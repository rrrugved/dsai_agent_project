FROM python:3.11-slim

WORKDIR /app

# Runtime tools used by Whisper and pytesseract.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg tesseract-ocr tini \
    && rm -rf /var/lib/apt/lists/*

# Keep the virtual environment in the project and make its commands available.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    LANGSMITH_TRACING=true \
    LANGSMITH_PROJECT=dsai-agent-project \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv

# Install the exact locked, production dependency set first. This layer is
# reused unless the project manifest or lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the application only after dependencies have been installed, then
# install the project itself without changing the lockfile.
COPY . ./
RUN uv sync --frozen --no-dev \
    && chmod +x /app/start.sh

EXPOSE 10000
EXPOSE 8000

ENTRYPOINT ["tini", "--"]
CMD ["/app/start.sh"]
