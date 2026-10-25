FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KNEE_HOST=0.0.0.0 KNEE_ALLOW_UPLOADS=0 HOME=/home/researcher XDG_CACHE_HOME=/opt/knee/.runtime/cache
WORKDIR /opt/knee
COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir --disable-pip-version-check --timeout 30 --retries 1 torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir --disable-pip-version-check --timeout 30 --retries 1 -r requirements-lock.txt && \
    python -m pip check && \
    python -c "import torch, torchvision; from pydicom.pixels import get_decoder; assert torch.__version__ == '2.11.0+cpu'; assert torchvision.__version__ == '0.26.0+cpu'; assert all(get_decoder(u).is_available for u in ['1.2.840.10008.1.2.4.70', '1.2.840.10008.1.2.4.90'])"
RUN useradd --create-home --uid 1000 researcher
COPY --chown=researcher:researcher app ./app
COPY --chown=researcher:researcher run.py ./run.py
RUN mkdir models .runtime && chown researcher:researcher models .runtime
USER researcher
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('KNEE_PORT', '7860') + '/health/ready', timeout=3).close()"]
CMD ["python", "run.py"]
