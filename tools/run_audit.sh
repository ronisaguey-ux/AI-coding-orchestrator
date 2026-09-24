#!/bin/bash
# Pass-1 sweep of the helpotron full-field audit.
# Free OpenRouter lanes only: the webchat lanes (ds:8081, gm:8085) take minutes per
# send and the engine's per-call budget is 300s, so they time out on every audit
# prompt. They stay wired for interactive work; the bulk goes to the API lanes.
cd /home/roni/Roni_workspace/AI-coding-orchestrator/engine
export DEEPSEEK_API_BASE=http://127.0.0.1:8090/v1
export OPENROUTER_KEY_FILE=/home/roni-saguey/.config/orch/openrouter.token
export AUDIT_OUTPUT_DIR=/home/roni/Roni_workspace/audits_plans
export AUDIT_TASK_FILE=/home/roni/Roni_workspace/promptsfr/audit_prompt.md
export AUDIT_VERSION=2026_09_24
export AUDIT_TARGET_LABEL=helpotron
export AUDIT_NUM_PASSES=2
export AUDIT_CHAT_TIMEOUT=300
export AUDIT_README_FILE=/home/roni/Roni_workspace/helpotron/README.md
export AUDIT_GRAPH_FILE=/home/roni/Roni_workspace/helpotron/graphify-out/graph.json
export AUDIT_INCLUDE_FILES="$(paste -sd, /tmp/opencode/audit_scope.txt)"
# Five separate free models: two were not enough and the run spent most of its
# rounds sitting in per-model cooldown after a 429.
export OCULUS_MODEL_ALLOWLIST="or-nvidia/nemotron-3-ultra-550b-a55b:free,or-poolside/laguna-s-2.1:free,or-nvidia/nemotron-3-super-120b-a12b:free"
exec /usr/bin/python3 audit.py --limit 5
