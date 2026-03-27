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

# Debug: Show what was copied
RUN echo "=== Directory contents after copy ===" && \
    ls -la && \
    echo "=== Looking for model files ===" && \
    find . -name "*.keras" -type f -exec ls -lh {} \; && \
    echo "=== Checking if model file is valid ===" && \
    if [ -f plant_disease_model.keras ]; then \
        echo "Model file exists:" && \
        ls -lh plant_disease_model.keras && \
        file plant_disease_model.keras || echo "Cannot determine file type"; \
    else \
        echo "Model file NOT found!" && \
        echo "Available files:" && ls -la; \
    fi

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]