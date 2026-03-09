FROM python:3.11

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-por \
    libmagic1 \
    poppler-utils \
    p7zip-full \
    unar \
    libreoffice-writer \
    libreoffice-calc \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# Core deps (obrigatório)
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Heavy deps: docling, marker-pdf (opcional — falha não quebra container)
COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt || echo "WARNING: heavy deps failed, continuing without them"
# Install torch with ROCm support (AMD Radeon) instead of default CUDA
RUN pip install --no-cache-dir "torch==2.7.0+rocm6.3" "torchvision==0.22.0+rocm6.3" --index-url https://download.pytorch.org/whl/rocm6.3


# EasyOCR — usa o torch ROCm já instalado acima (não reinstala torch)
RUN pip install --no-cache-dir easyocr

# RapidOCR + onnxruntime — engine OCR leve para o docling (CPU, sem GPU necessária)
RUN pip install --no-cache-dir onnxruntime rapidocr-onnxruntime

COPY main.py .
COPY zip_recursive.py .

RUN mkdir -p /app/storage

EXPOSE 7000

ENV MAX_WORKERS=2
ENV LOG_LEVEL=INFO

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "1"]
