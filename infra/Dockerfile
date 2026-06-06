FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim

# Create the runtime user first so subsequent COPYs can chown to it directly,
# rather than running an extra chown layer after USER is set.
RUN useradd --create-home appuser

WORKDIR /app
COPY --from=builder /install /usr/local

# Application source — owned by appuser so the process can write to
# monitoring/reports/ at runtime without an extra chown. The bulky / sensitive
# data trees (data/raw, data/processed, mlruns) are excluded via .dockerignore
# so the image stays small and applicant-level data is not baked in.
# data/processed/features.parquet is mounted read-only at runtime via
# docker-compose for the demo lookup path.
COPY --chown=appuser:appuser config/ config/
COPY --chown=appuser:appuser data/ data/
COPY --chown=appuser:appuser features/ features/
COPY --chown=appuser:appuser training/ training/
COPY --chown=appuser:appuser evaluation/ evaluation/
COPY --chown=appuser:appuser decisioning/ decisioning/
COPY --chown=appuser:appuser monitoring/ monitoring/
COPY --chown=appuser:appuser serving/ serving/
COPY --chown=appuser:appuser governance/ governance/
COPY --chown=appuser:appuser artifacts/ artifacts/

USER appuser

EXPOSE 8000

# /health returns 503 when MODEL_PATH does not resolve to a valid bundle;
# urllib.request.urlopen raises HTTPError on non-2xx so the probe exits
# non-zero and Docker marks the container unhealthy. The explicit timeout
# bounds socket-level hangs inside the probe.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
