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

# Dual-stack bind: Railway healthcheck hits IPv4, private DNS (*.railway.internal)
# resolves to AAAA only — need both. gunicorn supports multiple --bind; a single
# uvicorn --host :: is unreliable (kernel-level IPV6_V6ONLY varies per container).
CMD ["sh", "-c", "gunicorn main:app -k uvicorn.workers.UvicornWorker -w 1 --bind 0.0.0.0:${PORT:-8000} --bind [::]:${PORT:-8000}"]
