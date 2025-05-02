FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY credentials.json ./credentials.json

CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
