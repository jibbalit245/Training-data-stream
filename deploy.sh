#!/usr/bin/env bash
# deploy.sh
# One-command setup for the distributed data extraction pipeline.
# Works on RunPod CPU instances, HF Spaces, and local Linux/macOS.
#
# Usage:
#   bash deploy.sh              # full setup + launch pipeline
#   bash deploy.sh --dry-run    # setup only, skip pipeline launch
#   bash deploy.sh --test-only  # run validation tests then exit

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
DRY_RUN=false; TEST_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--dry-run"   ]] && DRY_RUN=true
  [[ "$arg" == "--test-only" ]] && TEST_ONLY=true
done

# ---------------------------------------------------------------------------
# 1. Python version check (3.9+)
# ---------------------------------------------------------------------------
info "Checking Python version (3.11+ required)…"
PYTHON=${PYTHON:-python3}
PY_VERSION=$("$PYTHON" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null) \
  || error "Python 3 not found. Install Python 3.9+ and retry."
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11 ) ]]; then
  error "Python >= 3.11 required (found $PY_VERSION)."
fi
info "Python $PY_VERSION OK"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment at $VENV_DIR…"
  "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
info "Virtual environment activated"

# ---------------------------------------------------------------------------
# 3. Install Python dependencies
# ---------------------------------------------------------------------------
info "Installing dependencies from requirements.txt…"
pip install --upgrade pip -q
pip install -r requirements.txt -q
info "Dependencies installed"

# ---------------------------------------------------------------------------
# 4. Create .env if missing
# ---------------------------------------------------------------------------
if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp .env.example .env
    warn ".env created from .env.example — edit it before running!"
  else
    cat > .env << 'ENVEOF'
# ── Hugging Face ───────────────────────────────────────────────────────────
HF_TOKEN=your_hf_write_token_here
HF_REPO_ID=your_username/reasoning-corpus
HF_SPLIT=train
HF_CHUNK_SIZE_MB=50

# ── Pipeline ───────────────────────────────────────────────────────────────
NUM_AGENTS=10
DEDUP_THRESHOLD=0.92
DEDUP_MODEL=all-MiniLM-L6-v2
DRY_RUN=false

# ── Retry ──────────────────────────────────────────────────────────────────
EXTRACTOR_MAX_RETRIES=5
EXTRACTOR_RETRY_MIN_WAIT=2
EXTRACTOR_RETRY_MAX_WAIT=60

# ── Database ───────────────────────────────────────────────────────────────
DB_PATH=pipeline.db

# ── Source data (comma-separated URLs) ────────────────────────────────────
DARWIN_URLS=
EINSTEIN_URLS=
PAULI_URLS=
PLATO_URLS=
DIALOGUE_URLS=
PDF_URLS=
ARXIV_QUERY=reasoning OR chain-of-thought OR argumentation
ARXIV_MAX_RESULTS=500
GITHUB_REPOS=huggingface/transformers,openai/openai-python
GITHUB_TOKEN=
SE_SITES=math,physics,philosophy,cs
SE_MIN_SCORE=10
SE_MAX_RESULTS=200
SE_API_KEY=
ENVEOF
    warn ".env created with placeholders — edit it before running the pipeline!"
  fi
fi

# Source .env so HF vars are available for the repo-creation step
set -o allexport
# shellcheck source=.env
source .env 2>/dev/null || true
set +o allexport

# ---------------------------------------------------------------------------
# 5. Create HF dataset repo (if credentials present and not dry-run/test-only)
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "false" && "$TEST_ONLY" == "false" ]]; then
  if [[ -n "${HF_REPO_ID:-}" && -n "${HF_TOKEN:-}" && "${HF_TOKEN}" != "your_hf_write_token_here" ]]; then
    info "Ensuring HF dataset repo ${HF_REPO_ID} exists…"
    python - << 'PYEOF'
import os, sys
try:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(
        repo_id=os.environ["HF_REPO_ID"],
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )
    print(f"  HF repo ready: {os.environ['HF_REPO_ID']}")
except Exception as e:
    print(f"  [warn] Could not create HF repo: {e}", file=sys.stderr)
PYEOF
  else
    warn "HF credentials not configured — skipping repo creation."
  fi
fi

# ---------------------------------------------------------------------------
# 6. Run validation tests
# ---------------------------------------------------------------------------
info "Running validation tests…"
if command -v pytest &>/dev/null; then
  pytest tests/ -v --tb=short 2>&1 \
    && info "All tests passed ✓" \
    || warn "Some tests failed — review output above."
else
  warn "pytest not installed — skipping tests (run: pip install pytest)"
fi

if [[ "$TEST_ONLY" == "true" ]]; then
  info "Test-only mode complete."
  exit 0
fi

# ---------------------------------------------------------------------------
# 7. Launch pipeline (unless --dry-run)
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
  info "Dry-run mode: pipeline launch skipped."
  info "To launch manually: source $VENV_DIR/bin/activate && python main.py"
else
  info "Launching pipeline (NUM_AGENTS=${NUM_AGENTS:-10})…"
  python main.py
fi

info "deploy.sh complete."
