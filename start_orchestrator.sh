#!/usr/bin/env bash
# Start (or resume) the orchestrator.
# Usage:
#   ./start_orchestrator.sh                 # resume from last state, start at execution phase
#   ./start_orchestrator.sh --phase audit   # start the full loop at audit
#   ./start_orchestrator.sh --status        # show orchestrator status and exit
set -euo pipefail

ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${ORCH_DIR}/.." && pwd)}"
VENV_PY="${VENV_PY:-$(command -v python3)}"

# Load secrets (gitignored .env) if present
ORCH_ENV="${ORCH_ENV:-${HOME}/.config/orchestrator/orchestrator.env}"
if [ -f "${ORCH_ENV}" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ORCH_ENV}"
  set +a
fi

# Give the pipeline scripts the paths they need
export REPO_DIR="${REPO_DIR}"
export WORK_DIR="${WORK_DIR:-${REPO_DIR}/../orchestrator_data}"
mkdir -p "${WORK_DIR}"

# Pipeline scripts are spawned as subprocesses — put the venv first on PATH
# so they resolve to the same interpreter that has the deps installed.
if [ -n "${VENV_DIR:-}" ]; then
  export PATH="${VENV_DIR}/bin:${PATH}"
fi
export LLM_API_KEY="${LLM_API_KEY:-}"
export PYTHONFAULTHANDLER=1

# ACTIVE PLAN: the executor defaults to the latest master_plan_*.md in
# ${WORK_DIR}. Pin a specific plan file here whenever one lands.
export MASTER_PLAN_FILE="${MASTER_PLAN_FILE:-}"
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"

EXTRA=("$@")
if [ ${#EXTRA[@]} -eq 0 ]; then
  EXTRA=(--phase execution --resume)
fi

echo "Starting orchestrator with: ${EXTRA[*]}"
exec "${VENV_PY}" "${ORCH_DIR}/run_workflow.py" --config "${ORCH_DIR}/config_workflow.yaml" "${EXTRA[@]}" 1>>"${WORK_DIR}/orchestrator_stdout.log" 2>>"${WORK_DIR}/orchestrator.log"
