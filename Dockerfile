# --- STAGE 1: API Service (EXPOSES 8000) ---
FROM python:3.11-slim as api-stage

WORKDIR /app

# Install build dependencies temporarily for installing some python packages
RUN apt-get update && apt-get install -y gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (FastAPI, uvicorn, pydantic, joblib, pandas)
COPY api/requirements.txt . # Assuming API requirements are separate
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./api ./api
COPY ./utils ./utils

# COPY ARTIFACTS: This is CRITICAL for the API service to load the models.
COPY ./artifacts ./artifacts

ENV PYTHONPATH=/app

EXPOSE 8000

# --- STAGE 2: STREAMLIT Service (EXPOSES 8501) ---
FROM python:3.11-slim as streamlit-stage

WORKDIR /app

# Install common system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Streamlit and dependencies, including 'requests' for API communication
COPY requirements.txt . # Assuming Streamlit requirements are here
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit app and supporting files
COPY streamlit_app.py .
COPY config.py .
COPY data/ ./data/
COPY reports/ ./reports/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]