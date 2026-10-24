FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KNEE_HOST=0.0.0.0 KNEE_ALLOW_UPLOADS=0
WORKDIR /opt/knee
COPY requirements*.txt ./
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-lock.txt
RUN useradd --create-home --uid 1000 researcher
COPY --chown=researcher:researcher app ./app
COPY --chown=researcher:researcher run.py ./run.py
RUN mkdir models .runtime && chown researcher:researcher models .runtime
USER researcher
EXPOSE 7860
CMD ["python", "run.py"]
