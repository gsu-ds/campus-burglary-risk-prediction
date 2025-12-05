# --- STAGE 1: API Service (EXPOSES 8000) ---
FROM python:3.11-slim as api-stage

WORKDIR /app

# Install build dependencies temporarily for installing some python packages
RUN apt-get update && apt-get install -y gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Install FastAPI dependencies
COPY api/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source code
COPY ./api ./api
# Note: Artifacts, reports, and utils are excluded here, relying on volume mounts.

ENV PYTHONPATH=/app

EXPOSE 8000

# --- STAGE 2: STREAMLIT Service (EXPOSES 8501) ---
FROM python:3.11-slim as streamlit-stage

WORKDIR /app

# Install common system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Streamlit dependencies (includes 'requests')
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit app and configuration file
# FIXED PATH: Copies from the subdirectory where streamlit_app.py is located.
COPY streamlit_app/streamlit_app.py .
# config.py is correctly copied from the root.
COPY config.py .

# Note: Data/reports are excluded here, relying on volume mounts.

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]