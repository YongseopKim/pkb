#!/usr/bin/env bash
#
# build.sh — Build Python package using .venv Python
#
# Usage:
#   ./scripts/build.sh            # Clean dist/ and build sdist + wheel
#   ./scripts/build.sh --no-clean # Build without cleaning dist/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist"

# ── Resolve Python (prefer .venv) ────────────────────────
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="python3"
    echo "Warning: .venv not found, falling back to system python3"
fi

# ── Parse args ───────────────────────────────────────────
NO_CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --no-clean) NO_CLEAN=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Read version ─────────────────────────────────────────
PYPROJECT="$REPO_ROOT/pyproject.toml"
version=$($PYTHON -c "
import re, pathlib
text = pathlib.Path('$PYPROJECT').read_text()
m = re.search(r'^version\s*=\s*\"([^\"]+)\"', text, re.MULTILINE)
print(m.group(1))
")

# ── Clean dist/ ──────────────────────────────────────────
if [ "$NO_CLEAN" = false ] && [ -d "$DIST_DIR" ]; then
    rm -rf "$DIST_DIR"
    echo "Cleaned dist/"
fi

# ── Build ────────────────────────────────────────────────
echo ""
echo "Building pkb $version ..."
$PYTHON -m build "$REPO_ROOT" --outdir "$DIST_DIR"

# ── Summary ──────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  Build complete: pkb $version"
echo "═══════════════════════════════════════════"
ls -lh "$DIST_DIR/"
echo ""
echo "Install with:"
echo "  pip install $DIST_DIR/pkb-${version}-py3-none-any.whl"
