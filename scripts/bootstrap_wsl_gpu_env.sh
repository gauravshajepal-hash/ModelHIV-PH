#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${EPIGRAPH_WSL_VENV:-$HOME/.venvs/modelhiv-ph-gpu}"
PYTHON_VERSION="${EPIGRAPH_WSL_PYTHON:-3.12}"
TORCH_INDEX_URL="${EPIGRAPH_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_TIMEOUT

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required inside WSL Ubuntu" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$(dirname "$VENV_DIR")"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

uv pip install --python "$VENV_DIR/bin/python" --index-url "$TORCH_INDEX_URL" torch
uv pip install --python "$VENV_DIR/bin/python" \
  gpytorch \
  "jax[cuda12]" \
  numpyro \
  pyarrow \
  scipy \
  pydantic \
  requests \
  polars \
  transformers \
  pytest

uv pip install --python "$VENV_DIR/bin/python" -e "$REPO_DIR"

echo
echo "WSL GPU environment is ready."
echo "Venv: $VENV_DIR"
echo "Repo: $REPO_DIR"
echo "Activate with: source \"$VENV_DIR/bin/activate\""
