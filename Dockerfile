# Use official Python slim image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps for librosa (librosa needs libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Expose port
ENV PORT=8080

# Command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
