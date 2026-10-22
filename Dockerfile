# Acquirer engine in a container: the same Python and package versions everywhere.
#
#   docker build -t acquirer-engine .
#   docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/runs:/app/runs" acquirer-engine
#   docker run --rm -v "$PWD/runs:/app/runs" acquirer-engine run --mode replay   # no key
#
# Reports are written to /app/runs; the -v option maps that to ./runs on your machine.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Pinned dependencies first: this layer is reused until requirements.lock changes.
COPY requirements.lock .
RUN pip install --root-user-action=ignore -r requirements.lock

# Then the engine itself (its dependencies are already installed above).
COPY pyproject.toml README.md config.yaml ./
COPY src ./src
COPY data ./data
RUN pip install --root-user-action=ignore --no-deps .

ENTRYPOINT ["acquirer-engine"]
CMD ["run"]
