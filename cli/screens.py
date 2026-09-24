"""screens.py — the orch control panel.

A screen is a function that draws once and returns a value:
    * a string  -> navigate to that screen
    * None      -> redraw in place
    * BACK      -> go up one level
    * raises QuitError -> leave the CLI

Keeping the contract this small is what stops a TUI becoming a state machine nobody can
follow: no screen knows about any other, and the menu is the only routing table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import ansi as A
from . import config, runs

BACK = "__back__"
W = lambda: A.term_width()


def wait_key(msg: str = "press any key") -> str:
    A.style
    print("\n" + A.dim("  " + msg))
    return A.read_key(None)


def menu(title: str, items: list[tuple], subtitle: str = "", extra: list[str] | None = None) -> str:
    """A vertical menu. items = [(label, hint, value)]. Returns the chosen value or BACK."""
    sel = 0
    while True:
        for line in extra or []:
            print(line)
        if subtitle:
            print("  " + A.dim(subtitle))
        print("  " + A.b(title))
        print("  " + A.dim("─" * (W() - 4)))
        for i, (label, hint, _v) in enumerate(items):
            if i == sel:
                print("  " + A.style("❯ " + label, "cyan", "bold") + ("  " + A.dim(hint) if hint else ""))
            else:
                print("    " + label + ("  " + A.dim(hint) if hint else ""))
        print()
        print("  " + A.dim("↑/↓ move   enter select   esc back   ctrl-c quit"))
        k = A.read_key(None)
        if k == "up":
            sel = (sel - 1) % max(1, len(items))
        elif k == "down":
            sel = (sel + 1) % max(1, len(items))
        elif k == "enter":
            return items[sel][2]
        elif k in ("escape", "q", "left"):
            return BACK
        elif k == "ctrl-c":
            raise A.QuitError()
        # clear and redraw
        print(A.clear(), end="")


# ── dashboard ────────────────────────────────────────────────────────────────

def screen_dashboard() -> str | None:
    p = runs.progress()
    cfg = {s["id"]: s for s in config.resolved_all()}
    lines = []

    agg = runs.aggregate_health()
    lines.append(("Endpoint", A.green("up") + " " + A.dim(agg.get("url", "")) if agg["ok"]
                  else A.red("down") + " " + A.dim(agg.get("error", ""))))
    lines.append(("Target", A.b(config.active_target() or "(none)")
                  + A.dim("   " + str((config.target_get(config.active_target()) or {}).get("dir", "")))))

    if p.get("run"):
        run = p["run"]
        state = A.green("running") if p["running"] else A.dim("stopped")
        lines.append(("Run", state + A.dim(f"  pid {run.get('pid')}  {run.get('target')}")))
        cb = p.get("currentBatch")
        if cb:
            lines.append(("Batch", f"pass {cb['pass']}  {cb['index']}/{cb['total']}  " + A.dim(cb["file"])))
        lines.append(("Rounds", f"{p.get('roundsOk', 0)}/{p.get('roundsTotal', 0)} ok   "
                                 f"{A.b(str(p.get('findings', 0)))} findings"))
        on_disk = runs.batches_on_disk()
        if on_disk:
            parts = [f"{k} {v['files']}f/{v['findings']}✱" for k, v in on_disk.items()]
            lines.append(("On disk", A.dim("  ".join(parts))))
    else:
        lines.append(("Run", A.dim("no runs yet")))

    width = max((A.visible_width(f"{k}") for k, _ in lines), default=10)
    body = [f"{A.dim(k.ljust(width))}  {v}" for k, v in lines]

    print(A.clear(), end="")
    print()
    print("  " + A.b("orch") + A.dim("  —  audit orchestrator"))
    print()
    print(A.box("Status", body, W() - 4))
    print()

    rows = p.get("rounds") or []
    if rows:
        rlines = []
        for r in rows[-8:]:
            mark = A.green("OK") if r["verdict"].upper() in ("OK", "PASS") else A.red(r["verdict"])
            rlines.append(f"{mark}  r{r['round']:<2} {r['agent'][:26]:<26} {r['seconds']:>6.1f}s  "
                          f"{r['findings']} findings")
        print(A.box("Recent rounds", rlines, W() - 4, "blue"))
    elif p.get("tail"):
        print(A.box("Log", [A.dim(t[:W() - 8]) for t in p["tail"]], W() - 4, "blue"))
    else:
        print(A.box("Recent rounds", [A.dim("waiting for the first round…")], W() - 4, "blue"))

    if p.get("running"):
        print()
        print("  " + A.dim("(refreshes on any keypress — enter to refresh now)"))
        k = A.read_key(3000)
        if k == "ctrl-c":
            raise A.QuitError()
        return None
    print()
    return "main"


def screen_main() -> str:
    while True:
        items = [
            ("Dashboard", "live status of the current run", "dashboard"),
            ("Start a run", "pick a target, audit it", "run"),
            ("Targets", "the repos this orchestrator audits", "targets"),
            ("Configuration", "every setting", "config"),
            ("Lanes & models", "which backends are alive", "models"),
            ("Findings", "what the last run produced", "findings"),
            ("Agent personas", "who audits, and what each one looks for", "agents"),
            ("MCP server", "let an agent drive this orchestrator", "mcp"),
        ]
        choice = menu("What do you want to do?", items)
        if choice == BACK:
            raise A.QuitError()
        return choice


# ── run ──────────────────────────────────────────────────────────────────────

def screen_run() -> str:
    while True:
        tnames = list(config.targets())
        if not tnames:
            print(A.clear(), end="")
            print("\n  " + A.red("No targets configured.") + "  Add one first.")
            return "targets"
        items = []
        for n in tnames:
            t = config.target_get(n) or {}
            why = runs._validate(n)
            hint = ("ready" if not why else why[0][:60])
            label = (A.green("✓ ") if not why else A.red("! ")) + n
            items.append((label, hint, n))
        items.append((A.dim("← back"), "", BACK))

        print(A.clear(), end="")
        print()
        choice = menu("Start an audit", items,
                      subtitle=f"output: {config.resolve('outputDir')[0]}   "
                               f"passes: {config.resolve('passes')[0]}   "
                               f"batch: {config.resolve('batchSize')[0]}   "
                               f"models: {config.resolve('modelAllowlist')[0]}")
        if choice == BACK:
            return "main"
        t = choice
        problems = runs._validate(t)
        print(A.clear(), end="")
        print()
        if problems:
            print(A.box("Cannot run " + t, [A.red(p) for p in problems] + ["", "Fix these on the Targets screen."], W() - 4, "red"))
            wait_key()
            return "targets"

        cur = runs.current_run()
        if cur and cur.get("alive"):
            print(A.box("A run is already going",
                        [f"pid {cur.get('pid')}  target {cur.get('target')}",
                         "", "Stop it first, or wait for it to finish."], W() - 4, "yellow"))
            wait_key()
            return "dashboard"

        print(A.box("About to start",
                    [f"target      {t}",
                     f"repo        {(config.target_get(t) or {}).get('dir')}",
                     f"passes      {config.resolve('passes')[0]}",
                     f"batch size  {config.resolve('batchSize')[0]} files",
                     f"agents      {config.resolve('concurrency')[0]} concurrent",
                     f"resume      {config.resolve('resume')[0]}",
                     f"models      {config.resolve('modelAllowlist')[0]}",
                     f"output      {config.resolve('outputDir')[0]}"],
                    W() - 4))
        print()
        if not A.confirm("  Start it?"):
            return "main"
        res = runs.start(t)
        print()
        if res["ok"]:
            r = res["run"]
            print(A.green(f"  ✓ started  pid {r['pid']}  log {r['log']}"))
            print(A.dim("  the panel can be closed; the run keeps going"))
        else:
            print(A.red("  ✗ " + str(res.get("reason"))))
        wait_key()
        return "dashboard"


# ── targets ──────────────────────────────────────────────────────────────────

def screen_targets() -> str:
    while True:
        tnames = list(config.targets())
        active = config.active_target()
        items = []
        for n in tnames:
            t = config.target_get(n) or {}
            dot = A.green("● ") if n == active else "  "
            items.append((dot + n, str(t.get("dir", ""))[:64], n))
        items.append((A.cyan("+ add a target"), "", "__add__"))
        items.append((A.dim("← back"), "", BACK))

        print(A.clear(), end="")
        print()
        choice = menu("Targets", items, subtitle="the repos this orchestrator audits")
        if choice == BACK:
            return "main"
        if choice == "__add__":
            name = A.prompt_line("\n  name for the new target (e.g. myrepo)").strip()
            if not name:
                continue
            d = A.prompt_line("  repository root").strip()
            if not d:
                continue
            res = config.target_save(name, {"dir": d, "label": name})
            print(A.green("  ✓ added " + name) if res["ok"] else A.red("  ✗ " + str(res.get("reason"))))
            wait_key()
            continue
        screen_target(choice)
        return "targets"


def screen_target(name: str) -> None:
    while True:
        t = config.target_get(name) or {}
        rows = [[k, str(t.get(k, "") or "")[:70]] for k, _ in config.TARGET_FIELDS]
        items = [(A.b("Use this target"), "make it the active one", "__use__")]
        for k, help_text in config.TARGET_FIELDS:
            items.append((k, str(t.get(k, "") or "(empty)")[:60], k))
        items.append((A.red("Remove this target"), "", "__rm__"))
        items.append((A.dim("← back"), "", BACK))

        print(A.clear(), end="")
        print()
        is_active = config.active_target() == name
        choice = menu(f"Target: {name}" + ("   " + A.green("(active)") if is_active else ""), items)
        if choice == BACK:
            return
        if choice == "__use__":
            config.set_active_target(name)
            continue
        if choice == "__rm__":
            if A.confirm(f"  Remove target '{name}'?"):
                res = config.target_remove(name)
                print(A.green("  ✓ removed") if res["ok"] else A.red("  ✗ " + str(res.get("reason"))))
                wait_key()
                if res["ok"]:
                    return
            continue
        # edit one field
        spec = dict(config.TARGET_FIELDS).get(choice, "")
        print("\n  " + A.dim(spec))
        new = A.prompt_line(f"  {choice}", str(t.get(choice, "") or ""))
        res = config.target_save(name, {choice: new})
        if not res["ok"]:
            print(A.red("  ✗ " + str(res.get("reason"))))
            wait_key()


# ── config ───────────────────────────────────────────────────────────────────

def screen_config() -> str:
    while True:
        rows = config.resolved_all()
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(r["group"], []).append(r)
        items = []
        for g, rs in groups.items():
            items.append((A.b(g), f"{len(rs)} settings", "__g__" + g))
        items.append((A.dim("← back"), "", BACK))
        print(A.clear(), end="")
        print()
        shadowed = config.count_shadowed()
        sub = f"{len(rows)} settings" + (f"   {A.yellow(str(shadowed) + ' overridden by the environment')}" if shadowed else "")
        choice = menu("Configuration", items, subtitle=sub)
        if choice == BACK:
            return "main"
        if choice.startswith("__g__"):
            group = choice[5:]
            _screen_config_group(group, groups[group])


def _screen_config_group(group: str, rows: list[dict]) -> None:
    while True:
        items = []
        for r in rows:
            val = r["value"]
            shown = "••••••" if r["kind"] == "secret" and val else (json.dumps(val) if isinstance(val, list) else str(val))
            tag = "" if r["source"] == "file" else A.dim(f" ({r['source']})")
            items.append((r["id"], shown[:52] + tag, r["id"]))
        items.append((A.dim("← back"), "", BACK))
        print(A.clear(), end="")
        print()
        choice = menu(group, items)
        if choice == BACK:
            return
        spec = config.BY_ID.get(choice)
        if not spec:
            continue
        print("\n  " + A.dim(spec["help"]))
        new = A.prompt_line(f"  {spec['label']}")
        if new == "":
            res = config.setting_reset(choice)
            if res["ok"]:
                note = "  " + A.yellow(f"(also cleared {res.get('clearedEnv') and 'env var' or 'nothing in env'})") if res.get("clearedEnv") else ""
                print(A.green("  ✓ reset to default") + note)
            else:
                print(A.red("  ✗ " + str(res.get("reason"))))
            wait_key()
            continue
        res = config.setting_save(choice, new)
        if res["ok"]:
            print(A.green(f"  ✓ {choice} = {res['value']}"))
        else:
            print(A.red("  ✗ " + str(res.get("reason"))))
        wait_key()
        # refresh the operator rows for this group
        for i, r in enumerate(rows):
            if r["id"] == choice:
                v, s = config.resolve(choice)
                rows[i] = {**r, "value": v, "source": s}


# ── models / lanes ───────────────────────────────────────────────────────────

def screen_models() -> str:
    print(A.clear(), end="")
    print()
    agg = runs.aggregate_health()
    if agg["ok"]:
        lines = [A.green("up") + "  " + A.dim(str(agg.get("url"))), ""]
        lines += ["  " + m for m in (agg.get("models") or [])]
    else:
        lines = [A.red("down") + "  " + str(agg.get("error"))]
    print(A.box("Model endpoint", lines, W() - 4, "green" if agg["ok"] else "red"))
    print()

    lanes = []
    for l in runs.lane_health():
        h = l.get("health") or {}
        if l.get("ok"):
            live = A.green("browser") if h.get("browserAlive") else A.yellow("no browser")
            busy = f"  in-flight {h.get('outstandingMs', 0)}ms" if h.get("outstandingMs") else ""
            lanes.append(f"{A.b(l['lane']):<12} :{l['port']}  {live}{busy}")
        else:
            lanes.append(f"{A.b(l['lane']):<12} :{l['port']}  {A.red('unreachable')}  {A.dim(str(l.get('error'))[:40])}")
    print(A.box("Webchat lanes", lanes or ["(none reachable)"], W() - 4, "blue"))
    print()
    allow = config.resolve("modelAllowlist")[0]
    print(A.box("Allowlist", [f"the engine will use only: {A.b(str(allow))}",
                              A.dim("change it in Configuration → Models")], W() - 4))
    wait_key()
    return "main"


# ── findings ─────────────────────────────────────────────────────────────────

def screen_findings() -> str:
    while True:
        run = runs.current_run()
        if not run:
            print(A.clear(), end="")
            print("\n  " + A.dim("No run yet."))
            wait_key()
            return "main"
        fs = runs.findings(run)
        sev: dict[str, int] = {}
        for f in fs:
            s = str(f.get("severity") or f.get("criticality") or "?").upper()
            sev[s] = sev.get(s, 0) + 1
        order = sorted(sev.items(), key=lambda kv: -kv[1])
        head = "   ".join(f"{A.b(k)} {v}" for k, v in order) or "(none)"

        items = [(A.dim("← back"), "", BACK)]
        for f in fs[:200]:
            title = str(f.get("title") or f.get("issue") or f.get("finding") or "")[:70]
            s = str(f.get("severity") or f.get("criticality") or "")
            col = {"CRITICAL": "red", "HIGH": "yellow", "MEDIUM": "cyan"}.get(s.upper(), "grey")
            items.append(((A.style(s[:4].ljust(4), col) + " " + title), str(f.get("file") or "")[:36], f))
        print(A.clear(), end="")
        print()
        print(A.box(f"Findings — {run.get('target')}", [head,
                                                        A.dim(f"{len(fs)} shown from {run.get('outputDir')}")], W() - 4))
        print()
        choice = menu("Select one to read", items)
        if choice == BACK:
            return "main"
        if isinstance(choice, dict):
            _show_finding(choice)


def _show_finding(f: dict) -> None:
    print(A.clear(), end="")
    print()
    body = []
    for k in ("title", "severity", "criticality", "confidence", "file", "line_range", "category"):
        if f.get(k):
            body.append(f"{A.dim(k.ljust(12))} {f[k]}")
    for k in ("description", "detail", "issue", "why", "impact", "recommendation", "fix", "prove"):
        if f.get(k):
            body.append("")
            body.append(A.b(k))
            body.append(str(f[k])[:1500])
    print(A.box("Finding", body, W() - 4))
    wait_key()


# ── agents ───────────────────────────────────────────────────────────────────

def screen_agents() -> str:
    """Show the personas that will actually run for the active target."""
    import ast as _ast

    engine = Path(str(config.resolve("enginePy")[0]))
    ns: dict = {"os": __import__("os")}
    try:
        tree = _ast.parse(engine.read_text())
        keep = [n for n in tree.body
                if isinstance(n, (_ast.Assign, _ast.FunctionDef))
                and (getattr(n, "name", None) in ("AGENTS", "DOMAIN_PROFILES", "DOMAIN_AGENTS", "active_agents")
                     or (isinstance(n, _ast.Assign) and isinstance(n.targets[0], _ast.Name)
                         and n.targets[0].id in ("AGENTS", "DOMAIN_PROFILES", "DOMAIN_AGENTS")))]
        exec(compile(_ast.Module(body=keep, type_ignores=[]), "<engine>", "exec"), ns)
    except Exception as e:
        print(A.clear(), end="")
        print("\n  " + A.red("could not read the engine's personas: " + str(e)))
        wait_key()
        return "main"

    label = str((config.target_get(config.active_target()) or {}).get("label") or config.active_target())
    ns["TARGET_LABEL"] = label
    agents = ns["active_agents"]()
    generic = {a["name"] for a in ns["AGENTS"]}

    lines = []
    for i, a in enumerate(agents):
        tag = A.dim("(generic)") if a["name"] in generic else A.cyan("(specialist)")
        lines.append(f"{i+1}.  {A.b(a['name'])}  w{a['weight']}  {tag}")
    prof = (ns["DOMAIN_PROFILES"].get(label) or {})
    body = [A.dim(f"target label: {label}   ({len(agents)} agents per batch)"), ""] + lines
    print(A.clear(), end="")
    print()
    print(A.box("Agent personas", body, W() - 4))
    if prof:
        print()
        print(A.box("Domain context injected into every persona",
                    [A.dim("what")] + _wrap(prof.get("what", ""), W() - 8) +
                    ["", A.dim("not a risk here")] + _wrap(prof.get("not_risks", ""), W() - 8),
                    W() - 4, "magenta"))
    wait_key()
    return "main"


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]


# ── mcp ──────────────────────────────────────────────────────────────────────

def screen_mcp() -> str:
    print(A.clear(), end="")
    print()
    from . import mcp_server as m
    names = [t["name"] for t in m.TOOLS]
    print(A.box("MCP server", [
        f"{A.b(str(len(names)))} tools — any agent can drive this orchestrator",
        "",
        A.dim("stdio:  orch mcp"),
        A.dim("register it in your MCP client's config and point it at: orch mcp"),
    ], W() - 4, "green"))
    print()
    cols = 2
    per = (len(names) + cols - 1) // cols
    lines = []
    for i in range(per):
        left = names[i] if i < len(names) else ""
        right = names[i + per] if i + per < len(names) else ""
        lines.append(f"{left:<30} {right}")
    print(A.box("Tools", lines, W() - 4))
    wait_key()
    return "main"
