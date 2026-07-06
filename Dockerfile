FROM python:3.12-slim AS base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS train
COPY src/ src/
RUN python -m src.train

FROM base AS app

# Create a non-root user
RUN useradd -m -r appuser && \
    chown appuser /app

COPY --from=train --chown=appuser /app/models /app/models
COPY --chown=appuser src/ src/
COPY --chown=appuser app/ app/
COPY --chown=appuser run.sh .

# Switch to non-root user
USER appuser

ENV API_BASE_URL=http://localhost:8000

EXPOSE 8501 8000
CMD ["./run.sh"]
