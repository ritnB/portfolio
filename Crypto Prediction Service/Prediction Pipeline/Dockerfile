# Lightweight Python image for production
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install Python dependencies (prefer layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Run Flask API with Gunicorn (1 hour timeout to support long inference)
CMD ["gunicorn", "--timeout", "3600", "--bind", "0.0.0.0:8080", "main:app"]
