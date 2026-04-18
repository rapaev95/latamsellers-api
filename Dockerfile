FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full FastAPI app (main.py + v2/ package + migrations/)
COPY main.py .
COPY v2/ ./v2/
COPY migrations/ ./migrations/

# Railway injects $PORT; uvicorn binds to it.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
