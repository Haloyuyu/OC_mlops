# Le modèle est sérialisé par cloudpickle sous Python 3.12 : même version mineure obligatoire.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False

# libgomp1 : runtime OpenMP requis par LightGBM, absent de l'image slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY model/ model/
COPY api/ api/

# Le journal de production est écrit par un utilisateur non-root : le dossier doit
# lui appartenir. À monter en volume (`-v ./logs:/app/logs`) pour que les journaux
# survivent au conteneur et soient analysables depuis l'hôte.
RUN useradd --create-home api \
    && mkdir -p /app/logs \
    && chown api /app/logs
USER api
VOLUME ["/app/logs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
