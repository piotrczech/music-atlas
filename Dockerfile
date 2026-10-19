FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev
RUN useradd --create-home atlas
USER atlas
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "music_atlas.api:app", "--host", "0.0.0.0", "--port", "8000"]
