#!/bin/bash
# Restart the OpenAI-compatible aggregate on :8090 without disturbing anything else.
# A stale aggregate holds the old code for its whole life, so an edit here is inert
# until this runs — the same rule as the webchat gateway.
for p in $(pgrep -f 'aggregat[e].py'); do kill "$p" 2>/dev/null; done
sleep 1
cd /home/roni/Roni_workspace/AI-coding-orchestrator
exec /usr/bin/python3 aggregate.py
