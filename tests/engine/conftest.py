"""Point the engine test suite at the copy that actually runs.

The repo carries the engine TWICE:

  <repo>/engine/   — the live set. Self-contained: orch_config, orch_git,
                     orch_lanes, orch_queue, orch_verify all sit beside
                     execute.py, and `import execute` works there.
  <repo>/          — stale duplicates from before the move. `import execute`
                     FAILS here (ModuleNotFoundError: orch_config), so the root
                     execute.py cannot run at all.

Without this file the suite resolved `import execute` / `import orch_queue` to
the root copies, and `test_escalation_invariants.py` could not be collected at
all — the escalation-invariant safety net was dead while reporting nothing.
Putting engine/ first makes every test exercise the live modules.
"""
import os
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2] / "engine"
if not (ENGINE / "execute.py").exists():          # tests/ deployed beside the engine
    ENGINE = Path(__file__).resolve().parents[1]

# Real config, or the engine resolves paths from its own location and refuses.
os.environ.setdefault("ORCH_CONFIG", os.path.expanduser("~/.config/orch/orch.yaml"))

if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))
