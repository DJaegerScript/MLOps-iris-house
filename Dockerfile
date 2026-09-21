FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN groupadd --system --gid 10001 iris \
    && useradd --system --uid 10001 --gid iris --create-home --home-dir /home/iris iris

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --requirement requirements.txt

# Only source code and non-sensitive manifest metadata are copied into the image.
COPY app.py ./app.py
COPY src ./src
COPY artifacts ./artifacts

USER iris

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.address=0.0.0.0", "--server.port=8501"]
