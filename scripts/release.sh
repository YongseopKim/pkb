#!/usr/bin/env bash
#
# release.sh — Version bump + build workflow
#
# Usage:
#   ./scripts/release.sh patch    # 0.8.0 → 0.8.1
#   ./scripts/release.sh minor    # 0.8.0 → 0.9.0
#   ./scripts/release.sh major    # 0.8.0 → 1.0.0
#   ./scripts/release.sh          # build only (no version bump)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYPROJECT="$REPO_ROOT/pyproject.toml"
DIST_DIR="$REPO_ROOT/dist"

# ── Read current version ──────────────────────────────────
current_version=$(python3 -c "
import re, pathlib
text = pathlib.Path('$PYPROJECT').read_text()
m = re.search(r'^version\s*=\s*\"([^\"]+)\"', text, re.MULTILINE)
print(m.group(1))
")

echo "Current version: $current_version"

# ── Bump version (if requested) ───────────────────────────
BUMP="${1:-}"

if [ -n "$BUMP" ]; then
    new_version=$(python3 -c "
import sys
parts = '$current_version'.split('.')
if len(parts) != 3:
    print('Error: version must be MAJOR.MINOR.PATCH', file=sys.stderr)
    sys.exit(1)
major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
bump = '$BUMP'
if bump == 'major':
    major += 1; minor = 0; patch = 0
elif bump == 'minor':
    minor += 1; patch = 0
elif bump == 'patch':
    patch += 1
else:
    print(f'Error: unknown bump type: {bump} (use major/minor/patch)', file=sys.stderr)
    sys.exit(1)
print(f'{major}.{minor}.{patch}')
")

    echo "Bumping: $current_version → $new_version"

    # Update pyproject.toml
    python3 -c "
import pathlib, re
p = pathlib.Path('$PYPROJECT')
text = p.read_text()
text = re.sub(
    r'^(version\s*=\s*)\"[^\"]+\"',
    r'\g<1>\"$new_version\"',
    text,
    count=1,
    flags=re.MULTILINE,
)
p.write_text(text)
"
    echo "Updated $PYPROJECT"
else
    new_version="$current_version"
    echo "No bump requested, building current version."
fi

# ── Clean dist/ ───────────────────────────────────────────
if [ -d "$DIST_DIR" ]; then
    rm -rf "$DIST_DIR"
    echo "Cleaned dist/"
fi

# ── Build ─────────────────────────────────────────────────
echo ""
echo "Building pkb $new_version ..."
python3 -m build "$REPO_ROOT" --outdir "$DIST_DIR"

# ── Summary ───────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  Build complete: pkb $new_version"
echo "═══════════════════════════════════════════"
ls -lh "$DIST_DIR/"
echo ""
echo "Install with:"
echo "  pip install $DIST_DIR/pkb-${new_version}-py3-none-any.whl"
