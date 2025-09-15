FROM python:3.11-slim

# Install system dependencies (tesseract OCR and fonts)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       libgl1 \
       poppler-utils \
       build-essential \
       pkg-config \
       libjpeg62-turbo \
       libpng16-16 \
       libtiff5 \
       ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (leverages Docker layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py ./

# Expose port
EXPOSE 8000

# Security: expect OPENAI_API_KEY to be provided at runtime
ENV PYTHONUNBUFFERED=1

# Use Uvicorn to serve FastAPI
CMD ["python", "app.py"]

