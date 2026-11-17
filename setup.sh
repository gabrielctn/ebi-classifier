#!/usr/bin/env bash
set -euo pipefail

# Set up a separate development environment without altering the older .venv.
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

if [ "$(uname -s)" != Darwin ]; then
    echo "This development setup currently targets macOS. Use the Windows package on Windows."
    exit 1
fi

if [ -x "$PROJECT_ROOT/.ebi-runtime/uv" ] && "$PROJECT_ROOT/.ebi-runtime/uv" --version >/dev/null 2>&1; then
    UV="$PROJECT_ROOT/.ebi-runtime/uv"
elif command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
else
    echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and retry."
    exit 1
fi

export UV_NO_CONFIG=1
export UV_SYSTEM_CERTS=true
export UV_CACHE_DIR="$PROJECT_ROOT/.ebi-runtime/cache"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.ebi-runtime/managed-python"

case "$(uname -m)" in
    arm64)
        PYTHON_VERSION="3.14.10"
        PLATFORM="arm64"
        PYTHON_REQUEST="$PYTHON_VERSION"
        if [ -x "$PROJECT_ROOT/.ebi-runtime/python/bin/python3.14" ]; then
            PYTHON_REQUEST="$PROJECT_ROOT/.ebi-runtime/python/bin/python3.14"
        fi
        ;;
    x86_64)
        PYTHON_VERSION="3.11.18"
        PLATFORM="intel"
        PYTHON_REQUEST="$PYTHON_VERSION"
        ;;
    *) echo "Unsupported Mac processor."; exit 1 ;;
esac

ENV_DIR="$PROJECT_ROOT/.venv-secure"
if [ ! -x "$ENV_DIR/bin/python" ]; then
    "$UV" venv --python "$PYTHON_REQUEST" "$ENV_DIR"
fi

"$ENV_DIR/bin/python" - "$PYTHON_VERSION" <<'PY'
import sys
required = tuple(map(int, sys.argv[1].split('.')))
if sys.version_info[:2] != required[:2] or sys.version_info[:3] < required:
    sys.exit(f"The existing .venv-secure requires Python {sys.argv[1]} or a newer patch in that series.")
PY

MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if [ "$PLATFORM" = arm64 ] && [ "$MACOS_MAJOR" -ge 14 ]; then
    "$UV" pip install --python "$ENV_DIR/bin/python" -r development/requirements-macos-arm64.txt
else
    # Keep the tested inference constraints for older/Intel Macs while adding
    # the tools used by the training scripts.
    "$UV" pip install --python "$ENV_DIR/bin/python" -r "macos/requirements-$PLATFORM.txt" \
        pip tensorboard scikit-learn matplotlib optuna plotly kaleido
fi

echo "Environment ready. Activate with: source .venv-secure/bin/activate"
echo "The previous .venv was not changed."
