FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/models/hf PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install silma-tts runpod "numpy==1.26.4" && pip uninstall -y torchvision torchcodec || true
COPY handler.py /handler.py
# Bake every model download (SILMA weights, vocoder, tashkeel, normalizers) into the image: no cold downloads.
RUN python -c "from silma_tts.api import SilmaTTS; SilmaTTS(device='cpu')" || python -c "from silma_tts.api import SilmaTTS; SilmaTTS()"
CMD ["python", "-u", "/handler.py"]
