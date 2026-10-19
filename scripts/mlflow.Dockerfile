FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev --extra tracking
RUN useradd --create-home atlas && mkdir /tracking && chown atlas:atlas /tracking
USER atlas
WORKDIR /tracking
EXPOSE 5000
CMD ["/app/.venv/bin/mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////tracking/mlflow.db", "--artifacts-destination", "/tracking/artifacts"]
