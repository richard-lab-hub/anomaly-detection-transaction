FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs build-essential curl ca-certificates zstd gosu \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# Install Ollama CLI (used to talk to the ollama service from this container)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create non-root user and group (UID/GID overridable at build time)
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid ${APP_GID} appuser \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/bash appuser

WORKDIR /app

EXPOSE 8000

COPY --chown=appuser:appuser requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=appuser:appuser . .
RUN pip install -e .

COPY --chown=appuser:appuser scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R appuser:appuser /app

ENV PATH="/home/appuser/.local/bin:${PATH}" \
    OLLAMA_HOST=http://ollama:11434

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
