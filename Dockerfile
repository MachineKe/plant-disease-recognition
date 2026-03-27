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

# Verify model integrity and print hash for debugging
RUN echo "=== Model Verification ===" && \
    if [ -f plant_disease_model.keras ]; then \
        echo "✓ Model file found" && \
        ls -lh plant_disease_model.keras && \
        echo "File size: $(stat -c%s plant_disease_model.keras) bytes" && \
        echo "Expected size: 21833202 bytes" && \
        echo "Checking HDF5 integrity:" && \
        python3 -c "import h5py; f = h5py.File('plant_disease_model.keras', 'r'); print('✓ HDF5 file is valid'); print('Keys in file:', list(f.keys())); f.close()" && \
        echo "Model file is ready"; \
    else \
        echo "✗ Model file NOT found!" && exit 1; \
    fi

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]