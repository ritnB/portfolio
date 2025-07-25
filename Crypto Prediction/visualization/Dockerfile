FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expect credentials.json to be mounted at runtime
# Do NOT copy secrets into image
# Example runtime: -v $(pwd)/credentials.json:/app/credentials.json
# CMD expects main.py with app = Flask(...)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
