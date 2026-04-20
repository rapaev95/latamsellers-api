FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright Chromium for the position scraper (services/ml_scraper.py).
# --with-deps installs libnss3/libatk1.0-0/libxcomposite1/etc. Pinning the
# browser path keeps it predictable across Railway redeploys.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN python -m playwright install --with-deps chromium

# Copy the full FastAPI app (main.py + v2/ package + migrations/)
COPY main.py .
COPY v2/ ./v2/
COPY migrations/ ./migrations/

# Railway injects $PORT; uvicorn binds to it.
ENV PORT=8000
EXPOSE 8000

# Single IPv6 bind with kernel IPv4-mapping (container has net.ipv6.bindv6only=0).
# *.railway.internal resolves to AAAA only and Railway's healthcheck arrives as
# IPv4 → both are served by the same [::] socket. uvicorn --host :: was
# unreliable because uvicorn forces IPV6_V6ONLY=1; gunicorn does not.
CMD ["sh", "-c", "gunicorn main:app -k uvicorn.workers.UvicornWorker -w 1 --bind [::]:${PORT:-8000}"]
