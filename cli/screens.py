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
import sys
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
            ("Scope a run", "files, batching and a time estimate first", "scope"),
            ("Run control", "pause / stop / rerun the live run", "run_control"),
            ("Batches", "saved work — inspect or discard one", "batches"),
            ("Findings", "what the last run produced", "findings"),
            ("System prompt", "what an auditor is ACTUALLY told", "prompt"),
            ("Targets", "the repos this orchestrator audits", "targets"),
            ("Configuration", "every setting", "config"),
            ("Lanes & models", "which backends are alive", "models"),
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



# ── scope: what a run would cover, before starting it ────────────────────────

def screen_scope() -> str:
    """Files, batching and a time estimate — the numbers to look at BEFORE a run."""
    target = config.active_target()
    print(A.clear(), end="")
    print()
    # A bare "scanning…" on a repo with hundreds of files is indistinguishable from a hang.
    # Show the count climbing, on one line, so the user can see it is working.
    def progress(n: int) -> None:
        sys.stdout.write("\r  " + A.dim(f"scanning… {n} files found"))
        sys.stdout.flush()
    print("  " + A.dim("scanning…"))
    scan = runs.scan_target(target, on_progress=progress)
    sys.stdout.write("\r" + " " * 50 + "\r")
    est = runs.estimate(target)
    is_active = config.active_target() == target

    body = []
    if scan.get("error"):
        body.append(A.red(str(scan["error"])))
    else:
        body.append(f"{A.dim('target'.ljust(12))} {A.b(target)}")
        body.append(f"{A.dim('files'.ljust(12))} {A.b(str(scan.get('files')))}   {A.dim('(' + str(scan.get('source')) + ')')}")
        body.append(f"{A.dim('batches'.ljust(12))} {est.get('batches')}   {A.dim(str(est.get('batchSize')) + ' files each')}")
        body.append(f"{A.dim('rounds'.ljust(12))} {est.get('totalRounds')}   {A.dim('across ' + str(est.get('passes')) + ' pass(es)')}")
        body.append(f"{A.dim('estimate'.ljust(12))} {A.b(str(est.get('estimatedHours')) + ' hours')}   "
                    f"{A.dim(str(est.get('avgRoundSeconds')) + 's/round')}")
        body.append(f"{A.dim('basis'.ljust(12))} {A.dim(str(est.get('basis'))[:60])}")
        if scan.get("missing"):
            body.append("")
            body.append(A.red(f"{len(scan['missing'])} listed file(s) do not exist:"))
            for m in scan["missing"][:5]:
                body.append(A.red("  " + str(m)))
    print(A.box("Scope of a run", body, W() - 4))

    if not scan.get("error") and scan.get("sample"):
        print()
        print(A.box("First files", [A.dim(f[:W() - 8]) for f in scan["sample"][:10]], W() - 4, "blue"))

    items = [("Edit the target's file list", "scope is set on the Targets screen", "targets"),
             (A.cyan("Start this run now"), "", "run"),
             (A.dim("← back"), "", BACK)]
    print()
    choice = menu("Next", items, subtitle="the estimate assumes every round succeeds at the average pace")
    if choice == BACK:
        return "main"
    return choice


# ── batches: inspect and discard saved work ──────────────────────────────────

def screen_batches() -> str:
    while True:
        bs = runs.batch_files()
        items = [(A.dim("← back"), "", BACK)]
        if not bs:
            print(A.clear(), end="")
            print()
            print(A.box("Batches", [A.dim("nothing saved yet"),
                                    "",
                                    A.dim("a batch is written only when ALL of its rounds finish")], W() - 4))
            wait_key()
            return "main"
        for b in bs:
            mark = A.green("✓") if b["findings"] else A.yellow("·")
            items.append((f"{mark} {b['pass']}/{b['batch']}",
                          f"{b['findings']} findings   {b['bytes']}B", b))
        print(A.clear(), end="")
        print()
        print(A.box("Saved batches", [A.dim("select one to discard it, so the next resumed run re-audits it")], W() - 4))
        print()
        choice = menu("Batches", items)
        if choice == BACK:
            return "main"
        show_batch(choice)


def show_batch(b: dict) -> None:
    print(A.clear(), end="")
    print()
    body = [f"{A.dim('pass'.ljust(10))} {b['pass']}",
            f"{A.dim('batch'.ljust(10))} {b['batch']}",
            f"{A.dim('findings'.ljust(10))} {b['findings']}",
            f"{A.dim('size'.ljust(10))} {b['bytes']} bytes",
            "", A.b("files covered")]
    body += [A.dim("  " + str(f)[:W() - 12]) for f in b["files"]]
    print(A.box("Batch", body, W() - 4))
    print()
    items = [(A.red("Discard this batch"), "the next resumed run re-audits it", "@del"),
             (A.dim("← back"), "", BACK)]
    choice = menu("Action", items)
    if choice == "@del":
        if A.confirm("  Discard it? The file is moved to .discarded, not deleted", default=True):
            r = runs.delete_batch(None, b["pass"], b["batch"])
            print(A.green("  ✓ discarded") if r.get("ok") else A.red("  ✗ " + str(r.get("reason"))))
            wait_key()


# ── prompt preview: what an auditor is actually told ─────────────────────────

def screen_prompt() -> str:
    """The exact system prompt a persona receives. A model auditing a system it was told is
    something else invents defects belonging to that system, and this is how you see it."""
    from . import mcp_server as m
    try:
        r = m._prompt_preview({"target": config.active_target()})
    except Exception as e:
        print(A.clear(), end="")
        print("\n  " + A.red("could not build the prompt: " + str(e)))
        wait_key()
        return "main"
    if isinstance(r, dict) and r.get("isError"):
        print(A.clear(), end="")
        print("\n  " + A.red(r["content"][0]["text"]))
        wait_key()
        return "main"

    text = r["prompt"]
    # page through it: a real prompt is several thousand characters
    pos = 0
    page = max(10, A.term_height() - 8)
    while True:
        print(A.clear(), end="")
        print()
        print(A.box(f"{r['agent']}  ·  {r['chars']} chars  ·  "
                    f"domain context: {'yes' if r['domainContextIncluded'] else 'NO'}",
                    [], W() - 4, "magenta"))
        for line in text.splitlines()[pos:pos + page]:
            print("  " + line[:W() - 4])
        print()
        pct = min(100, int(100 * (pos + page) / max(1, len(text.splitlines()))))
        print("  " + A.dim(f"{pct}%   ↑/↓ scroll   space next page   esc back"))
        k = A.read_key(None)
        if k in ("escape", "q"):
            return "main"
        if k == "down":
            pos = min(max(0, len(text.splitlines()) - 1), pos + 1)
        elif k == "up":
            pos = max(0, pos - 1)
        elif k in ("space", "pagedown"):
            pos = min(max(0, len(text.splitlines()) - 1), pos + page)
        elif k in ("pageup",):
            pos = max(0, pos - page)


# ── run control: pause / resume / stop the live run ──────────────────────────

def screen_run_control() -> str:
    run = runs.current_run()
    if not run:
        print(A.clear(), end="")
        print("\n  " + A.dim("no run to control"))
        wait_key()
        return "main"
    while True:
        p = runs.progress(run)
        alive = p.get("running")
        body = [f"{A.dim('target'.ljust(11))} {run.get('target')}",
                f"{A.dim('pid'.ljust(11))} {run.get('pid')}   {A.green('alive') if alive else A.dim('stopped')}",
                f"{A.dim('rounds'.ljust(11))} {p.get('roundsOk')}/{p.get('roundsTotal')}   "
                f"{A.b(str(p.get('findings')))} findings",
                f"{A.dim('started'.ljust(11))} {time.strftime('%H:%M:%S', time.localtime(run.get('startedAt') or 0))}"]
        print(A.clear(), end="")
        print()
        print(A.box("Run control", body, W() - 4))
        print()
        if not alive:
            print("  " + A.dim("this run has finished"))
            wait_key()
            return "main"
        # "@" marks a LOCAL ACTION rather than a view to route to. Without a marker the two
        # are indistinguishable in the data, and code (or a reader) cannot tell that "stop"
        # is meant to run something here rather than navigate to a screen called stop.
        items = [("@Pause", "SIGSTOP — keeps its place and memory, uses no CPU", "@pause"),
                 ("@Stop", "terminates it; saved batches are kept", "@stop"),
                 ("@Rerun", "stop this and start a fresh one", "@rerun"),
                 (A.dim("← back"), "", BACK)]
        choice = menu("Action", items)
        if choice == BACK:
            return "main"
        if choice == "@pause":
            r = runs.pause(run["id"])
            print(A.green("  ✓ paused") if r.get("ok") else A.red("  ✗ " + str(r.get("reason"))))
            wait_key()
            # offer resume immediately, since a paused run looks identical to a dead one
            if r.get("ok") and A.confirm("  Resume it now?", default=True):
                runs.resume(run["id"])
        elif choice == "@stop":
            if A.confirm("  Stop the run?", default=True):
                r = runs.stop(run["id"])
                print(A.green("  ✓ stopped") if r.get("stopped") else A.dim("  " + str(r.get("reason"))))
                wait_key()
                return "main"
        elif choice == "@rerun":
            if A.confirm("  Stop and start fresh?", default=False):
                r = runs.rerun(run.get("target"))
                print(A.green("  ✓ started pid " + str(r["run"]["pid"])) if r.get("ok")
                      else A.red("  ✗ " + str(r.get("reason"))))
                wait_key()
                return "dashboard"


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
