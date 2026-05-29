#!/usr/bin/env bash
# Builds standalone, double-clickable executables for the three tools
# using PyInstaller. Run this once from the project root and three
# binaries land in dist/: pynozzle-moc2d, pynozzle-stt, pynozzle-moc3d.
#
# Usage:
#   ./build_executables.sh
#
# Requirements: Ubuntu (or any Linux with Python 3.9+) and an internet
# connection to fetch PyInstaller the first time.

set -euo pipefail
cd "$(dirname "$0")"

# 1. Make a venv if one isn't already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -d .venv ]]; then
        echo ">>> Creating virtualenv in .venv"
        python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# 2. Install the package + PyInstaller
echo ">>> Installing pynozzle + PyInstaller"
pip install -q --upgrade pip
pip install -q -e .
pip install -q pyinstaller

# 3. Build each tool as a single-file executable
mkdir -p dist build
rm -rf build pynozzle-*.spec

for tool in moc2d stt moc3d; do
    name="pynozzle-${tool}"
    echo ">>> Building ${name}"

    # Tiny launcher script that calls the CLI's main() function.
    cat > "/tmp/${name}_launcher.py" <<EOF
from pynozzle.${tool}.cli import main
import sys
sys.exit(main())
EOF

    pyinstaller \
        --onefile \
        --name "${name}" \
        --distpath dist \
        --workpath "build/${name}" \
        --specpath build \
        --collect-all pynozzle \
        --hidden-import scipy._cyutility \
        --hidden-import scipy.special._cdflib \
        "/tmp/${name}_launcher.py" \
        > "build/${name}.log" 2>&1

    chmod +x "dist/${name}"
    echo "    -> dist/${name}  ($(du -h "dist/${name}" | cut -f1))"
done

echo
echo "Done. Run them by double-clicking in your file manager, or:"
echo "  ./dist/pynozzle-moc2d  examples/M3.5Perf.inp           -o out/"
echo "  ./dist/pynozzle-stt    examples/stt/M3.5Perf.inp -i examples/stt -o stt_out/"
echo "  ./dist/pynozzle-moc3d  examples/moc3d/cone10.geo  --z-step 1     -o moc3d_out/"
