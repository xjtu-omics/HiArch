#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTDIR="${OUTDIR:-Output}"
VENV_DIR="${VENV_DIR:-.venv-hiarch}"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

rm -rf "$OUTDIR"
python scripts/balance_mcool.py
python scripts/prepare_hiarch_from_mcool.py
python scripts/write_hiarch_parameters.py
bash publish/HiArch/one_click_pipeline.sh "$OUTDIR/parameters.txt"
