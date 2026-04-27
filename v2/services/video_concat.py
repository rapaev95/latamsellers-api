"""Video concat service — склейка нескольких видео в одно через ffmpeg.

Use case: Runway даёт максимум 10с за один запрос. Для FB Reels нужно
15-30с. Генерим N клипов параллельно через Runway, потом склеиваем
здесь и отдаём один MP4.

ffmpeg concat demuxer (-f concat) — самый быстрый путь без перекодирования
если все клипы одинакового формата (Runway всегда отдаёт одно и то же:
H.264 + AAC + одинаковое разрешение/fps). При несовпадении формата —
fallback на concat filter с перекодированием.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 60.0
FFMPEG_TIMEOUT = 180  # 3 минуты — generous для 30с клипов
MAX_VIDEOS = 5
MAX_DOWNLOAD_SIZE_MB = 200


class VideoConcatError(Exception):
    """Bubble up cause for the API layer."""


async def _download(http: httpx.AsyncClient, url: str, dest: Path) -> None:
    """Stream-скачать видео в dest. Лимит размера чтобы не выжрать диск."""
    async with http.stream("GET", url, timeout=DOWNLOAD_TIMEOUT) as r:
        if r.status_code != 200:
            raise VideoConcatError(f"download_{r.status_code}: {url}")
        size = 0
        with dest.open("wb") as f:
            async for chunk in r.aiter_bytes(chunk_size=64 * 1024):
                size += len(chunk)
                if size > MAX_DOWNLOAD_SIZE_MB * 1024 * 1024:
                    raise VideoConcatError(f"file_too_large: {url} > {MAX_DOWNLOAD_SIZE_MB}MB")
                f.write(chunk)


def _run_ffmpeg(args: list[str]) -> tuple[int, str]:
    """sync wrapper над subprocess. Возвращает (returncode, stderr)."""
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT,
            check=False,
        )
        return proc.returncode, proc.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return -1, "ffmpeg_timeout"
    except FileNotFoundError:
        return -1, "ffmpeg_not_installed"


async def concat_videos(urls: list[str]) -> bytes:
    """Скачать N видео по URL, склеить через ffmpeg, вернуть итоговый MP4.

    Raises VideoConcatError при любом фейле (download/encode/timeout).
    """
    if not urls:
        raise VideoConcatError("no_urls")
    if len(urls) > MAX_VIDEOS:
        raise VideoConcatError(f"too_many_urls: max={MAX_VIDEOS}")
    if shutil.which("ffmpeg") is None:
        raise VideoConcatError("ffmpeg_not_installed_on_server")

    tmpdir = Path(tempfile.mkdtemp(prefix="vconcat_"))
    try:
        # 1) Скачать все клипы параллельно
        async with httpx.AsyncClient() as http:
            tasks = []
            paths: list[Path] = []
            for i, url in enumerate(urls):
                p = tmpdir / f"input_{i:02d}.mp4"
                paths.append(p)
                tasks.append(_download(http, url, p))
            await asyncio.gather(*tasks)

        # 2) Сделать concat list для ffmpeg demuxer
        listfile = tmpdir / "list.txt"
        with listfile.open("w") as f:
            for p in paths:
                # ffmpeg concat: каждая строка `file '/abs/path.mp4'`
                f.write(f"file '{p.as_posix()}'\n")

        output = tmpdir / "concat.mp4"

        # 3) Попытка №1 — concat demuxer без перекодирования (мгновенно)
        rc, err = _run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output),
        ])
        if rc != 0 or not output.exists() or output.stat().st_size == 0:
            log.warning("ffmpeg concat-copy failed (rc=%s): %s", rc, err[:500])
            # 4) Fallback — concat filter с перекодированием (для разных форматов)
            output.unlink(missing_ok=True)
            input_args: list[str] = []
            for p in paths:
                input_args.extend(["-i", str(p)])
            n = len(paths)
            filter_complex = (
                "".join(f"[{i}:v:0][{i}:a:0]" if i < n else "" for i in range(n))
                + f"concat=n={n}:v=1:a=1[outv][outa]"
            )
            # Видео может быть БЕЗ аудио (Runway), тогда без [i:a:0]
            filter_complex_noaudio = (
                "".join(f"[{i}:v:0]" for i in range(n))
                + f"concat=n={n}:v=1:a=0[outv]"
            )
            rc2, err2 = _run_ffmpeg([
                "ffmpeg", "-y",
                *input_args,
                "-filter_complex", filter_complex_noaudio,
                "-map", "[outv]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output),
            ])
            if rc2 != 0 or not output.exists():
                raise VideoConcatError(f"ffmpeg_failed: rc={rc2} err={err2[:300]}")

        return output.read_bytes()
    finally:
        # Чистим temp независимо от исхода
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
