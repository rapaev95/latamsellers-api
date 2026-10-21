FROM python:3.11-slim

WORKDIR /app

# ffmpeg для video concat (склейка Runway-клипов в FB Reels длиннее 10с)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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
# --timeout 120 (> the 90s internal _COMPUTE_TIMEOUT_SECONDS): a cold report
# compute for a large project (ARTHUR) can run tens of seconds; gunicorn's
# default 30s timeout was killing the worker mid-compute, so the proxy saw
# UND_ERR_SOCKET and the bundle never reached the durable cache — making every
# load a fresh, doomed recompute. The raised timeout lets the compute finish,
# cache, and serve ~50ms thereafter.
# Recycle the worker periodically. Threads that a timed-out compute abandoned
# can never be reclaimed — Python cannot kill a thread — so they accumulate
# until pthread_create fails and EVERY sync endpoint 500s with
# "RuntimeError: can't start new thread". That took ~2 weeks of uptime to hit
# on 2026-10-21. Recycling resets the count; it treats the symptom, not the
# leak.
#
# The number is deliberately high. The APScheduler lives INSIDE this worker,
# so every recycle restarts all cron jobs and will kill a long one mid-run —
# the nightly cache warm can run 15 minutes. Rare recycles are the point.
# /health reports threads and requests served, so tune this from real numbers
# rather than guessing: pick a value the process reaches about once a day.
CMD ["sh", "-c", "gunicorn main:app -k uvicorn.workers.UvicornWorker -w 1 --timeout 120 --graceful-timeout 120 --max-requests 5000 --max-requests-jitter 500 --bind [::]:${PORT:-8000}"]
