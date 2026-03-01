# syntax=docker/dockerfile:1

# Minimal image to build and run ctrl-freeq
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install any lightweight system deps if needed by wheels (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only what is necessary for installation to maximize layer caching
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

# Install the package (and its runtime dependencies) into the image
RUN python -m pip install --upgrade pip && \
    pip install .

# Default command is non-GUI to work in headless CI environments
# Prints installed version to verify the container runs
CMD ["python", "-c", "import ctrl_freeq; print('ctrl_freeq', ctrl_freeq.__version__)"]
