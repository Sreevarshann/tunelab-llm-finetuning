# Linux containers cannot use the Apple GPU: MLX runs on its CPU backend here (mlx[cpu]).
# Training and evaluation are real but much slower than native Metal; see docs/DEPLOYMENT.md.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 HF_HOME=/data/hf TUNELAB_DATA_DIR=/data/tunelab
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev && useradd -m -u 1000 app && mkdir -p /data && chown app /data
USER app
EXPOSE 8114
CMD ["/app/.venv/bin/uvicorn", "tunelab.main:app", "--host", "0.0.0.0", "--port", "8114"]
