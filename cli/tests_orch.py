#!/usr/bin/env python3
"""Tests for the orch CLI and MCP.

Run:  python3 cli/tests_orch.py

These check the invariants that are easy to break and expensive to notice:

  * the scan's exclusion lists MATCH the engine's. A scan that disagrees with the engine
    reported 4,109 files for a repo the engine audits ~840 of, which turns a usable estimate
    into an alarming one and misleads anyone planning a run.
  * every MCP tool is reachable through tools/call. A tool appended after TOOL_MAP was built
    is advertised by tools/list and then refused as "unknown tool" — the shape of a bug that
    reads like a client problem.
  * every setting resolves and round-trips, so "I changed it and nothing happened" cannot be
    caused by a typo in the schema.
  * the panel's menu values all exist as screens, or the CLI routes to nothing and exits.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

FAILS = []
PASSES = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSES
    if cond:
        PASSES += 1
        print(f"  ok   {name}")
    else:
        FAILS.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")


def test_excludes_match_engine():
    """The scan's exclusion lists must equal the engine's, or the estimate lies."""
    engine = ROOT / "engine" / "audit.py"
    tree = ast.parse(engine.read_text())
    ns: dict = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in ("EXCLUDE_DIRS", "EXCLUDE_FILES"):
                ns[n.targets[0].id] = ast.literal_eval(n.value)
    runs = importlib.import_module("cli.runs")
    check("EXCLUDE_DIRS matches the engine",
          ns.get("EXCLUDE_DIRS") == runs.DEFAULT_EXCLUDES,
          f"engine={len(ns.get('EXCLUDE_DIRS') or [])} cli={len(runs.DEFAULT_EXCLUDES)} "
          f"diff={sorted(set(ns.get('EXCLUDE_DIRS') or []) ^ runs.DEFAULT_EXCLUDES)[:6]}")
    check("EXCLUDE_FILES matches the engine",
          ns.get("EXCLUDE_FILES") == runs.DEFAULT_EXCLUDE_FILES,
          f"diff={sorted(set(ns.get('EXCLUDE_FILES') or []) ^ runs.DEFAULT_EXCLUDE_FILES)[:6]}")


def test_tool_map_complete():
    from cli import mcp_server as m
    names = [t["name"] for t in m.TOOLS]
    check("tool names are unique", len(names) == len(set(names)),
          f"duplicates: {[n for n in names if names.count(n) > 1][:5]}")
    missing = [n for n in names if n not in m.TOOL_MAP]
    check("every advertised tool is callable", not missing, f"unreachable: {missing[:5]}")
    check("tool count is at least 40", len(names) >= 40, f"only {len(names)}")
    bad = [t["name"] for t in m.TOOLS
           if not t.get("description") or not isinstance(t.get("inputSchema"), dict)
           or not callable(t.get("handler"))]
    check("every tool has a description, a schema and a handler", not bad, f"bad: {bad[:5]}")

    # required params must exist as properties, or a caller cannot satisfy them
    broken = []
    for t in m.TOOLS:
        props = set((t.get("inputSchema") or {}).get("properties") or {})
        for req in (t.get("inputSchema") or {}).get("required") or []:
            if req not in props:
                broken.append(f"{t['name']}.{req}")
    check("every required parameter is declared", not broken, f"undeclared: {broken[:5]}")


def test_settings_round_trip():
    from cli import config
    paths = [s["id"] for s in config.SCHEMA]
    check("settings are unique", len(paths) == len(set(paths)),
          f"dupes: {[p for p in paths if paths.count(p) > 1][:4]}")
    # every setting resolves to (value, source) without raising
    bad = []
    for p in paths:
        try:
            v, src = config.resolve(p)
            if src not in ("file", "env", "default"):
                bad.append(f"{p} source={src}")
        except Exception as e:
            bad.append(f"{p}: {e}")
    check("every setting resolves", not bad, f"{bad[:4]}")

    # a bad value must be REFUSED, not stored. Use a scratch config so the real one is safe.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.environ["ORCH_CONFIG"] = str(Path(td) / "c.json")
        importlib.reload(config)
        r = config.setting_save("passes", "9999")
        check("an out-of-range value is refused", not r["ok"], f"got {r}")
        r2 = config.setting_save("passes", "3")
        check("a valid value saves", r2["ok"] and r2["value"] == 3, f"got {r2}")
        v, src = config.resolve("passes")
        check("a saved value reads back from the file", v == 3 and src == "file", f"{v} {src}")
        r3 = config.setting_reset("passes")
        v2, src2 = config.resolve("passes")
        check("reset returns the default", r3["ok"] and src2 == "default", f"{v2} {src2}")
        r4 = config.setting_save("nope.not.real", "x")
        check("an unknown setting is refused", not r4["ok"], f"got {r4}")
    os.environ.pop("ORCH_CONFIG", None)
    importlib.reload(config)


def test_panel_routes_exist():
    """Every value a menu can return must be a screen the panel knows."""
    from cli import screens
    known = {n for n in dir(screens) if n.startswith("screen_")}
    src = Path(screens.__file__).read_text()
    # the routing table lives in cli/index.py
    idx = (HERE / "index.py").read_text()
    routed = set()
    for line in idx.splitlines():
        line = line.strip()
        if line.startswith("elif view ==") or line.startswith("if view =="):
            parts = line.split('"')
            if len(parts) >= 2:
                routed.add(parts[1])
    missing = [v for v in routed if f"screen_{v}" not in known and v != "main"]
    check("every routed view has a screen function", not missing, f"missing: {missing}")
    # And the menu must not offer a view the router does not know.
    # Match only a LINE THAT IS a menu-item tuple. A looser pattern also matches the key
    # handler `elif k in ("escape", "q", "left"):` and reports "left" as an unroutable view —
    # a test finding its own subject rather than the code's behaviour.
    offered = set()
    for line in (HERE / "screens.py").read_text().splitlines():
        st = line.strip()
        if not st.startswith("("):
            continue
        m = __import__("re").match(r'\("[^"]+",\s*"[^"]*",\s*"([a-zA-Z_]+)"\)', st)
        if m:
            offered.add(m.group(1))
    # A value beginning with "@" is a LOCAL ACTION, not a view to route to; the other
    # reserved values are handled inline by the screen that raised them.
    reserved = {"__add__", "__use__", "__rm__"}
    unr = [o for o in offered
           if o not in routed and o not in reserved and not o.startswith("@")]
    check("the menu offers only routable views", not unr, f"unroutable: {unr}")


def test_cli_help_and_subcommands():
    """The entry point must answer --help and refuse an unknown command clearly."""
    orch = ROOT / "orch"
    r = subprocess.run([str(orch), "help"], capture_output=True, text=True, timeout=120)
    check("orch help exits 0", r.returncode == 0, f"exit {r.returncode}")
    check("help lists the new commands", "orch estimate" in r.stdout and "orch batches" in r.stdout,
          "missing from usage")
    r2 = subprocess.run([str(orch), "definitely-not-a-command"], capture_output=True, text=True, timeout=120)
    check("an unknown command exits non-zero", r2.returncode != 0, f"exit {r2.returncode}")


def test_unreadable_config_is_reported_and_preserved():
    """A typo in the config file must not be silent, and must not be destructive.

    Measured before the fix: one bad character made every setting appear to revert (defaults
    were substituted with no warning, exit 0), and the next `config set` replaced the whole
    file with defaults — a custom value present before the save was gone after it.
    """
    import importlib
    import tempfile
    from cli import config
    real = config.CONFIG_FILE
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "c.json"
        os.environ["ORCH_CONFIG"] = str(cf)
        importlib.reload(config)
        # healthy first
        config.save_raw({**config._defaults(), "passes": 3}, cf)
        check("a readable config reports no problem", config.config_problem() is None, "")
        # break it, keeping a value we can look for
        cf.write_text('{"passes": 3,}')
        prob = config.config_problem()
        check("an unreadable config is reported", bool(prob) and "error" in (prob or {}),
              f"got {prob}")
        r = config.setting_save("passes", "4")
        check("saving over an unreadable config warns", bool(r.get("warning")), f"got {r}")
        bak = r.get("backedUp")
        check("saving over an unreadable config backs the original up",
              bool(bak) and Path(bak).exists(), f"backedUp={bak}")
        if bak and Path(bak).exists():
            check("the backup holds the original text",
                  '"passes"' in Path(bak).read_text(), "backup is empty or wrong")
        # repair restores a parseable file
        import shutil
        shutil.copy2(cf, Path(td) / "keep.json")
        data, _ = config.load_raw()
        config.save_raw(data, cf)
        check("a repaired config parses again", config.config_problem() is None, "")
    os.environ.pop("ORCH_CONFIG", None)
    importlib.reload(config)


def test_no_terminal_does_not_crash():
    """Prompts and key reads must degrade, not raise, when stdin is not a terminal.

    Measured before the fix: `orch config repair` fed from a pipe died with
    `termios.error: Inappropriate ioctl for device`, which makes every prompting command
    unusable in a script.
    """
    from cli import ansi
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.path.insert(0, %r);"
                        "from cli import ansi;"
                        "print('key=' + repr(ansi.read_key_safe(10)));"
                        "print('confirm=' + repr(ansi.confirm('q?', default=True)))" % str(ROOT)],
                       capture_output=True, text=True, input="y\n", timeout=60)
    check("read_key_safe returns without a terminal",
          r.returncode == 0 and "key=" in r.stdout, f"exit {r.returncode} {r.stderr[-120:]}")
    check("confirm reads a piped answer",
          r.returncode == 0 and "confirm=True" in r.stdout, f"stdout={r.stdout[-120:]}")
    # and the entry point explains itself instead of raising
    r2 = subprocess.run([str(ROOT / "orch")], capture_output=True, text=True,
                        stdin=subprocess.DEVNULL, timeout=60)
    check("the panel refuses without a terminal, without a traceback",
          "Traceback" not in r2.stderr and "needs a terminal" in (r2.stdout + r2.stderr),
          f"exit {r2.returncode} err={r2.stderr[-120:]}")


def test_empty_allowlist_is_explained():
    """An empty allowlist is VALID and means deny-all. Saving it must say so."""
    import importlib
    import tempfile
    from cli import config
    with tempfile.TemporaryDirectory() as td:
        os.environ["ORCH_CONFIG"] = str(Path(td) / "c.json")
        importlib.reload(config)
        r = config.setting_save("modelAllowlist", "")
        check("an empty allowlist saves (it is a legal value)", r["ok"], f"got {r}")
        check("an empty allowlist explains deny-all", bool(r.get("note")), f"got {r}")
    os.environ.pop("ORCH_CONFIG", None)
    importlib.reload(config)


def test_probe_result_shape():
    """A probe result must carry every key a caller reads, on EVERY path.

    An early return with a shorter dict produced `KeyError: 'elapsedMs'` in the CLI, turning a
    clear error message into a traceback.
    """
    from cli import runs
    # force the no-model path by asking for a probe with an empty allowlist
    import importlib
    import tempfile
    from cli import config
    with tempfile.TemporaryDirectory() as td:
        os.environ["ORCH_CONFIG"] = str(Path(td) / "c.json")
        importlib.reload(config)
        importlib.reload(runs)
        config.setting_save("modelAllowlist", "")
        r = runs.endpoint_probe(timeout=5)
        for k in ("ok", "model", "elapsedMs", "error", "url"):
            check(f"probe result carries '{k}'", k in r, f"keys={sorted(r)}")
        check("the no-model path does not claim success", r["ok"] is False, str(r))
        check("it names the cause", "allowlist" in str(r.get("error", "")), str(r.get("error"))[:80])
    os.environ.pop("ORCH_CONFIG", None)
    importlib.reload(config)
    importlib.reload(runs)


def main() -> int:
    print("orch tests\n")
    test_excludes_match_engine()
    test_tool_map_complete()
    test_settings_round_trip()
    test_panel_routes_exist()
    test_cli_help_and_subcommands()
    test_unreadable_config_is_reported_and_preserved()
    test_no_terminal_does_not_crash()
    test_empty_allowlist_is_explained()
    test_probe_result_shape()
    print(f"\n  {PASSES} passed, {len(FAILS)} failed")
    for f in FAILS:
        print("   ✗ " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
