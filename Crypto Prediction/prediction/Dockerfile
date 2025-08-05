FROM python:3.10-slim

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Flask + gunicorn entrypoint
CMD ["gunicorn", "--timeout", "3600", "--bind", "0.0.0.0:8080", "app:app"]
