FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch wheels keep the image small; torchvision follows the same index.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src
ENV DIFFUSION_CHECKPOINT=/app/models/ema.pt

EXPOSE 8000

CMD ["uvicorn", "diffusion.app:app", "--host", "0.0.0.0", "--port", "8000"]
