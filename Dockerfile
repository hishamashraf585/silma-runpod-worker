FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/models/hf PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
# CUDA build of PyTorch for the GPU worker (the local CPU env uses the CPU build of the same stack)
RUN pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple
# Exact versions frozen from the working local SILMA env: no resolver backtracking, no surprises.
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
COPY handler.py /handler.py
# Bake every model download (SILMA weights, vocoder, tashkeel, normalizers) into the image: no cold downloads.
RUN python -c "from silma_tts.api import SilmaTTS; SilmaTTS(device='cpu'); print('models baked')"
CMD ["python", "-u", "/handler.py"]
