"""NF (Nota Fiscal) parser — XML (preferred, structured) + DANFE PDF fallback.

NF-e XML (Brazilian electronic invoice) structure:
  <nfeProc>
    <NFe>
      <infNFe>
        <ide>nNF, dhEmi, natOp, ...</ide>
        <emit>CNPJ, xNome, enderEmit</emit>
        <dest>CNPJ, xNome</dest>
        <det nItem="N">
          <prod>cProd, xProd, NCM, CFOP, qCom, vUnCom, vProd</prod>
          <imposto>...</imposto>
        </det>
        ...
        <total><ICMSTot>vBC, vICMS, vProd, vNF</ICMSTot></total>
        <infAdic><infCpl>...DI...II...AFRMM...SISCOMEX...</infCpl></infAdic>
      </infNFe>
    </NFe>
  </nfeProc>

We extract a flat dict for the wizard.

DANFE PDF is the human-readable rendering of the same XML. Used only as a
fallback when the customer doesn't have the XML — text extraction is
fragile (column alignment varies). PDF parsing thresholds:
- Up to ~15 lines: pdfplumber usually fine
- More: require XML
"""
from __future__ import annotations

import io
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

# NF-e XML default namespace.
NFE_NS = "http://www.portalfiscal.inf.br/nfe"
NS = {"n": NFE_NS}


# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS nf_uploads (
  id              SERIAL PRIMARY KEY,
  ls_user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  filename        TEXT,
  content_type    TEXT,             -- 'application/xml' | 'application/pdf'
  source_format   TEXT NOT NULL,    -- 'xml' | 'pdf'
  size_bytes      INTEGER,
  parsed_json     JSONB NOT NULL,   -- ParsedNF as dict
  raw_text        TEXT,             -- xml string OR pdf text dump (for re-parse)
  created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_nf_uploads_user
  ON nf_uploads(ls_user_id, created_at DESC);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ──────────────────────────────────────────────────────────────────────────────
# Data shapes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NFLine:
    n_item: int
    sku: Optional[str]
    description: str
    ncm: Optional[str]
    cfop: Optional[str]
    unit: Optional[str]          # uCom — PAR/PC/UN…
    quantity: float
    unit_cost: float
    total_cost: float
    ean: Optional[str] = None    # cEAN if not «SEM GTIN»


@dataclass
class NFEmitter:
    cnpj: Optional[str]
    name: Optional[str]
    municipality: Optional[str] = None
    uf: Optional[str] = None


@dataclass
class NFImportCosts:
    di_number: Optional[str] = None
    ii_brl: Optional[float] = None       # Imposto de Importação
    afrmm_brl: Optional[float] = None
    siscomex_brl: Optional[float] = None
    is_import: bool = False              # «Compra de Mercadorias - Importação» natOp


@dataclass
class ParsedNF:
    nf_number: Optional[str]
    nf_series: Optional[str]
    nf_date: Optional[str]               # ISO date
    natural_op: Optional[str]            # natOp
    chave_acesso: Optional[str]
    emitter: NFEmitter
    destination_name: Optional[str]
    total_brl: Optional[float]
    lines: list[NFLine] = field(default_factory=list)
    import_costs: NFImportCosts = field(default_factory=NFImportCosts)


def parsed_to_dict(p: ParsedNF) -> dict[str, Any]:
    return asdict(p)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _txt(el: Optional[ET.Element], tag: str) -> Optional[str]:
    if el is None:
        return None
    found = el.find(f"n:{tag}", NS)
    return found.text.strip() if found is not None and found.text else None


def _num(el: Optional[ET.Element], tag: str) -> Optional[float]:
    val = _txt(el, tag)
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# XML parser
# ──────────────────────────────────────────────────────────────────────────────

class NFParseError(Exception):
    pass


def parse_nfe_xml(content: bytes | str) -> ParsedNF:
    """Parse NF-e XML. Accepts bytes or string. Tolerates either <nfeProc>
    (signed-and-authorized) or just <NFe> (locally generated)."""
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8")
        except UnicodeDecodeError:
            content = content.decode("latin-1")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as err:
        raise NFParseError(f"invalid_xml: {err}") from err

    # Strip namespace from root tag for matching
    tag_no_ns = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    if tag_no_ns == "nfeProc":
        nfe = root.find("n:NFe", NS)
        if nfe is None:
            raise NFParseError("missing_NFe_in_nfeProc")
    elif tag_no_ns == "NFe":
        nfe = root
    else:
        raise NFParseError(f"unsupported_root_tag: {tag_no_ns}")

    inf = nfe.find("n:infNFe", NS)
    if inf is None:
        raise NFParseError("missing_infNFe")

    chave = (inf.attrib.get("Id") or "").replace("NFe", "") or None

    ide = inf.find("n:ide", NS)
    emit = inf.find("n:emit", NS)
    dest = inf.find("n:dest", NS)
    total = inf.find("n:total", NS)
    info_adic = inf.find("n:infAdic", NS)

    nat_op = _txt(ide, "natOp")
    is_import = bool(nat_op and "import" in nat_op.lower())

    emitter = NFEmitter(
        cnpj=_txt(emit, "CNPJ"),
        name=_txt(emit, "xNome"),
        municipality=_txt(emit.find("n:enderEmit", NS) if emit is not None else None, "xMun"),
        uf=_txt(emit.find("n:enderEmit", NS) if emit is not None else None, "UF"),
    )

    icms_tot = total.find("n:ICMSTot", NS) if total is not None else None
    total_brl = _num(icms_tot, "vNF")

    # Date in ide.dhEmi is ISO with TZ. We trim to YYYY-MM-DD.
    dh_emi = _txt(ide, "dhEmi")
    nf_date: Optional[str] = None
    if dh_emi:
        try:
            nf_date = datetime.fromisoformat(dh_emi).date().isoformat()
        except ValueError:
            nf_date = dh_emi[:10]

    # Lines
    lines: list[NFLine] = []
    for det in inf.findall("n:det", NS):
        prod = det.find("n:prod", NS)
        if prod is None:
            continue
        n_item_raw = det.attrib.get("nItem")
        try:
            n_item = int(n_item_raw) if n_item_raw else len(lines) + 1
        except ValueError:
            n_item = len(lines) + 1

        ean_raw = _txt(prod, "cEAN")
        ean = ean_raw if ean_raw and ean_raw.upper() != "SEM GTIN" else None

        lines.append(NFLine(
            n_item=n_item,
            sku=_txt(prod, "cProd"),
            description=_txt(prod, "xProd") or "",
            ncm=_txt(prod, "NCM"),
            cfop=_txt(prod, "CFOP"),
            unit=_txt(prod, "uCom"),
            quantity=_num(prod, "qCom") or 0.0,
            unit_cost=_num(prod, "vUnCom") or 0.0,
            total_cost=_num(prod, "vProd") or 0.0,
            ean=ean,
        ))

    # Import costs from infAdic/infCpl (only for import NFs).
    import_costs = NFImportCosts(is_import=is_import)
    if info_adic is not None:
        cpl = _txt(info_adic, "infCpl") or ""
        if cpl:
            # DI: digits + slashes + dashes, but must end on a digit (avoid
            # capturing trailing period from «DI nº 26/0623840-0.»).
            di_match = re.search(
                r"DI\s*n[ºo]?\s*[:\.]?\s*([0-9][0-9./-]*[0-9])", cpl, re.IGNORECASE,
            )
            if di_match:
                import_costs.di_number = di_match.group(1)
            for label, attr in (
                ("II", "ii_brl"),
                ("AFRMM", "afrmm_brl"),
                ("SISCOMEX", "siscomex_brl"),
            ):
                # Word boundary so «II» doesn't match in «AFRMM», and
                # capture must end on a digit so trailing «,»/«.» from the
                # sentence is excluded.
                m = re.search(
                    rf"\b{label}\b[^\d]{{0,20}}([0-9][\d.,]*[0-9])",
                    cpl, re.IGNORECASE,
                )
                if m:
                    val = _parse_brl_number(m.group(1))
                    if val is not None:
                        setattr(import_costs, attr, val)

    return ParsedNF(
        nf_number=_txt(ide, "nNF"),
        nf_series=_txt(ide, "serie"),
        nf_date=nf_date,
        natural_op=nat_op,
        chave_acesso=chave,
        emitter=emitter,
        destination_name=_txt(dest, "xNome") if dest is not None else None,
        total_brl=total_brl,
        lines=lines,
        import_costs=import_costs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PDF parser (DANFE) — fallback only
# ──────────────────────────────────────────────────────────────────────────────

# Hard line-count cap above which we refuse the PDF and ask for XML — column
# layout in DANFE breaks at 15+ lines on most templates we've seen.
PDF_LINE_LIMIT = 15


def extract_pdf_text(file_bytes: bytes) -> str:
    """Pulls plain text out of a DANFE PDF. Order is page → block → line; line
    spans columns may interleave. We rely on `pdfplumber` for layout-aware
    extraction."""
    try:
        import pdfplumber
    except ImportError as err:
        raise NFParseError("pdfplumber_not_installed") from err

    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
    return "\n".join(parts)


def parse_danfe_pdf(file_bytes: bytes) -> ParsedNF:
    """Best-effort DANFE PDF parser. Use only when XML isn't available.

    Strategy: pull text, regex out the DANFE header fields (nNF, série, chave,
    total) + a line table. Lines look like:
        CFOP3102 Calcados Unisex ... {000001}  64041900 1900 3.102 PAR 210,00 7,7515 1.627,82 ...
    """
    text = extract_pdf_text(file_bytes)
    if not text.strip():
        raise NFParseError("pdf_no_text")

    # Header bits.
    nf_number_m = re.search(r"N[ºo]\s*([\d.]+)", text)
    serie_m = re.search(r"S[eé]rie\s*[:\.]?\s*(\d+)", text, re.IGNORECASE)
    chave_m = re.search(r"((?:\d{4}\s*){11})", text)  # 44-digit access key with spaces
    total_m = re.search(r"VALOR\s+TOTAL\s+DA\s+NOTA[^\d]{1,30}([\d.,]+)",
                        text, re.IGNORECASE)
    nat_op_m = re.search(r"NATUREZA\s+DA\s+OPERA[ÇC][ÃA]O\s*[:\n]*([^\n]+)",
                         text, re.IGNORECASE)
    nat_op_val = nat_op_m.group(1).strip() if nat_op_m else None
    is_import = bool(nat_op_val and "import" in nat_op_val.lower())

    total_brl: Optional[float] = None
    if total_m:
        raw = total_m.group(1).replace(".", "").replace(",", ".")
        try:
            total_brl = float(raw)
        except ValueError:
            pass

    # Lines — DANFE prod table row has 8+ numeric tokens at end after textual
    # description. We grep candidates by NCM regex (4 digits + optional 4).
    lines: list[NFLine] = []
    line_pattern = re.compile(
        # description tail + ncm + cst + cfop + unit + qty + vunit + vprod
        r"^([^\n]+?)\s+(\d{4}\d{4})\s+\d{3,4}\s+([\d.]+)\s+([A-Z]{1,4})\s+"
        r"([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)",
        re.MULTILINE,
    )
    for n_item, m in enumerate(line_pattern.finditer(text), start=1):
        desc = m.group(1).strip()
        ncm = m.group(2)
        cfop = m.group(3)
        unit = m.group(4)
        qty = _parse_brl_number(m.group(5))
        vunit = _parse_brl_number(m.group(6))
        vprod = _parse_brl_number(m.group(7))
        if qty is None or vprod is None:
            continue
        lines.append(NFLine(
            n_item=n_item, sku=None,
            description=desc, ncm=ncm, cfop=cfop, unit=unit,
            quantity=qty, unit_cost=vunit or 0.0, total_cost=vprod,
        ))
        if len(lines) > PDF_LINE_LIMIT:
            raise NFParseError(
                f"pdf_too_many_lines (>{PDF_LINE_LIMIT}); please upload XML"
            )

    # Import costs from observações block.
    import_costs = NFImportCosts(is_import=is_import)
    di_m = re.search(r"DI\s*n[ºo]?\s*[:\.]?\s*([0-9./-]+)", text, re.IGNORECASE)
    if di_m:
        import_costs.di_number = di_m.group(1)
    for label, attr in (("II", "ii_brl"), ("AFRMM", "afrmm_brl"),
                        ("SISCOMEX", "siscomex_brl")):
        m = re.search(rf"{label}\s*R?\$?\s*([\d.,]+)", text, re.IGNORECASE)
        if m:
            val = _parse_brl_number(m.group(1))
            if val is not None:
                setattr(import_costs, attr, val)

    return ParsedNF(
        nf_number=nf_number_m.group(1).replace(".", "") if nf_number_m else None,
        nf_series=serie_m.group(1) if serie_m else None,
        nf_date=None,        # Hard to disambiguate dates in DANFE — leave to user
        natural_op=nat_op_val,
        chave_acesso=re.sub(r"\s+", "", chave_m.group(1)) if chave_m else None,
        emitter=NFEmitter(cnpj=None, name=None),
        destination_name=None,
        total_brl=total_brl,
        lines=lines,
        import_costs=import_costs,
    )


def _parse_brl_number(s: str) -> Optional[float]:
    """`1.234,56` → 1234.56. Handles bare integers too."""
    s = s.strip()
    if not s:
        return None
    # BRL: dot=thousands, comma=decimal. Convert.
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Unified facade
# ──────────────────────────────────────────────────────────────────────────────

def detect_format(filename: str, content_type: Optional[str], peek: bytes) -> str:
    """Return 'xml' or 'pdf'; raise on neither."""
    lower_fname = (filename or "").lower()
    ct = (content_type or "").lower()
    if lower_fname.endswith(".xml") or "xml" in ct:
        return "xml"
    if lower_fname.endswith(".pdf") or "pdf" in ct or peek[:4] == b"%PDF":
        return "pdf"
    # Last-ditch: XML usually starts with '<?xml' or '<'
    head = peek[:64].lstrip()
    if head.startswith(b"<?xml") or head.startswith(b"<"):
        return "xml"
    raise NFParseError("unsupported_file_format")


def parse_nf(file_bytes: bytes, filename: str = "", content_type: Optional[str] = None
             ) -> tuple[ParsedNF, str]:
    """Returns (parsed, source_format)."""
    fmt = detect_format(filename, content_type, file_bytes[:512])
    if fmt == "xml":
        return parse_nfe_xml(file_bytes), "xml"
    return parse_danfe_pdf(file_bytes), "pdf"


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

async def save_upload(
    pool: asyncpg.Pool,
    ls_user_id: int,
    *,
    filename: str,
    content_type: Optional[str],
    source_format: str,
    parsed: ParsedNF,
    size_bytes: int,
    raw_text: Optional[str],
) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO nf_uploads (
                ls_user_id, filename, content_type, source_format,
                size_bytes, parsed_json, raw_text
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            RETURNING id
            """,
            ls_user_id, filename, content_type, source_format,
            size_bytes, json.dumps(parsed_to_dict(parsed)), raw_text,
        )
    return int(row["id"])
