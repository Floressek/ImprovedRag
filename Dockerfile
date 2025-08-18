# Minimal Dockerfile placeholder for RAG system
FROM python:3.11-slim

WORKDIR /app

# Install system deps if needed (left as comments for now)
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . ./

# Default command can be adjusted later
CMD ["python", "-c", "print('RAG container placeholder')"]
