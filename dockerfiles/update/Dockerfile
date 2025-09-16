FROM python:3.10-slim AS builder

ENV POETRY_VERSION=1.7.1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl build-essential git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock* ./
RUN poetry export --without-hashes --format=requirements.txt --output /tmp/requirements.txt --without dev

FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /tmp/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

COPY src/rankings /app/src/rankings

# Provide mount points for data/compiled volumes (matching compose defaults)
RUN mkdir -p /app/data /app/compiled

COPY dockerfiles/update/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

VOLUME ["/app/data", "/app/compiled"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
