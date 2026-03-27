FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY web_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and unzip the model
COPY model.zip .
RUN unzip model.zip && \
    rm model.zip

# Copy the rest of the web_app
COPY web_app/ .

# Verify model exists and is valid
RUN echo "=== Model Verification ===" && \
    ls -lh plant_disease_model.keras && \
    python3 -c "import h5py; f = h5py.File('plant_disease_model.keras', 'r'); print('✓ Model file is valid HDF5'); f.close()"

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]