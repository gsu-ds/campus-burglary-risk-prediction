FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Install deps first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app code
COPY ./api ./api
COPY ./streamlit_app.py .
COPY ./models ./models
COPY ./utils ./utils

ENV PYTHONPATH=/app

EXPOSE 8000 8501
