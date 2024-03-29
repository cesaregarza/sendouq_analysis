###############################
#         Base Image          #
###############################
ARG BASE_IMAGE=python:3.10-slim

FROM $BASE_IMAGE AS base

WORKDIR /app

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1
ENV PATH="$PATH:$POETRY_HOME/bin"

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - 

RUN poetry config virtualenvs.create false

###############################
#    Install  Dependencies    #
###############################
FROM base AS dependencies

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-dev

###############################
#        Build Image          #
###############################
FROM dependencies AS build

ARG BUILD_VERSION

COPY . /app/

# Build the application
RUN poetry version $BUILD_VERSION && \
    poetry build && \
    poetry install && \
    poetry update

EXPOSE 8050
