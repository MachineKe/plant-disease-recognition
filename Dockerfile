FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY web_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from web_app
COPY web_app/ .

# Verify model file exists
RUN echo "=== Model File Check ===" && \
    if [ -f plant_disease_model.keras ]; then \
        ls -lh plant_disease_model.keras && \
        echo "✓ Model file found"; \
    else \
        echo "✗ Model file NOT found!" && exit 1; \
    fi

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]