FROM python:3.10-slim

WORKDIR /app

# Install build dependencies for some Python packages (if needed)
RUN apt-get update && apt-get install -y gcc

# Copy requirements and install
COPY requirements_backend.txt .

RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy backend code
COPY . .

# Set env var pointing to secrets folder inside container (mount this folder at runtime)
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp_key.json

# Expose FastAPI port
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
