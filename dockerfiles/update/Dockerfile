FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VERSION=1.7.1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl build-essential git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

# Copy source code and then install the root package so console scripts are available
COPY src ./src
COPY scripts ./scripts
RUN poetry install --no-interaction --no-ansi

# Ensure local source is importable for python -m invocations
ENV PYTHONPATH=/app/src
COPY dockerfiles/update/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

VOLUME ["/app/data", "/app/compiled"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
