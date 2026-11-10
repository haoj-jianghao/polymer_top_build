#!/usr/bin/env bash
# Copy opls-ff-lite from its home in the opls-ff-gen repository into
# third_party/opls-ff-lite/. The OPLS/2020 SI files (data/opls2020/) are ACS
# material and are never copied (INTERFACE.md section 1).
#
#   scripts/sync_opls_ff_lite.sh [/path/to/OPLS_ff_gen/lite]
set -euo pipefail

SOURCE="${1:-${OPLS_FF_LITE_SOURCE:-$HOME/projects/OPLS_ff_gen/lite}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/third_party/opls-ff-lite"

if [[ ! -f "$SOURCE/src/opls_ff_lite/__init__.py" ]]; then
    echo "opls-ff-lite source not found at $SOURCE" >&2
    exit 1
fi

mkdir -p "$DEST/src"
rsync -a --delete \
    --exclude '__pycache__/' --exclude '*.py[cod]' \
    --exclude 'data/opls2020/' \
    "$SOURCE/src/opls_ff_lite/" "$DEST/src/opls_ff_lite/"
for file in LICENSE NOTICE README.md INTERFACE.md run_opls_ff_lite.py; do
    cp "$SOURCE/$file" "$DEST/$file"
done

commit="$(git -C "$SOURCE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
version="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$DEST/src/opls_ff_lite/__init__.py")"
cat > "$DEST/VENDORED.txt" <<INFO
opls-ff-lite $version, copied from the opls-ff-gen repository (lite/) at commit $commit
by scripts/sync_opls_ff_lite.sh. Do not edit these files here; change them
upstream and re-run the script. data/opls2020/ (OPLS/2020 SI) is not included:
each user installs it with
    python third_party/opls-ff-lite/run_opls_ff_lite.py install-si jp3c06602_si_002.txt jp3c06602_si_003.txt
INFO
echo "opls-ff-lite $version ($commit) -> $DEST"
