"""index.py — the `orch` command.

Two front ends in one binary:
  * no arguments (or `orch ui`) -> the interactive control panel.
  * a subcommand -> a plain one-shot command, so `orch` is scriptable and usable over ssh
    where a full-screen TUI is not.

Every subcommand prints and exits; the panel is for a human who wants to browse. That split
matters because a TUI that owns the terminal cannot also be piped.
"""

from __future__ import annotations

import json
import sys

from . import ansi as A
from . import config, runs

USAGE = """orch — audit orchestrator

  orch                      open the control panel
  orch run [target]         start an audit (detached; survives this shell)
  orch status               current run: pid, progress, findings
  orch watch                follow a run until it finishes
  orch stop                 stop the current run
  orch targets              list targets and whether each is ready
  orch findings [--severity S] [--limit N]
  orch config               list every setting with its source
  orch config get <id>
  orch config set <id> <value>
  orch config reset <id>
  orch config repair        recover from a config file that will not parse
  orch agents [target]      the personas that will run
  orch prompt [agent]       the EXACT system prompt a persona receives
  orch models               endpoint and lane health
  orch probe                time one request through the endpoint
  orch scan [target]        what a run WOULD audit (file count, batching)
  orch estimate [target]    rough wall-clock for a run
  orch batches              batches saved on disk, with their findings
  orch discard <pass> <b>   make the next resumed run re-audit one batch
  orch pause / orch resume  stop and continue a run without losing its place
  orch export [--format f]  write findings as json / md / csv
  orch logs                 every log this orchestrator has written
  orch doctor               check the environment before a run
  orch mcp                  run the MCP server on stdio (for an agent to drive)
  orch help
"""


def _out(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ── subcommands ──────────────────────────────────────────────────────────────

def cmd_status(argv: list[str]) -> int:
    p = runs.progress()
    run = p.get("run")
    if not run:
        A.style
        print(A.dim("no runs yet — `orch run <target>` to start one"))
        return 1
    state = A.green("running") if p["running"] else A.dim("stopped")
    print(f"{state}  pid {run.get('pid')}  target {A.b(str(run.get('target')))}")
    cb = p.get("currentBatch")
    if cb:
        print(f"  batch   pass {cb['pass']}  {cb['index']}/{cb['total']}  {cb['file']}")
    print(f"  rounds  {p.get('roundsOk', 0)}/{p.get('roundsTotal', 0)} ok   {p.get('findings', 0)} findings")
    disk = runs.batches_on_disk()
    if disk:
        print("  on disk " + "  ".join(f"{k}: {v['files']}f/{v['findings']}✱" for k, v in disk.items()))
    if p.get("lastLine"):
        print(A.dim("  " + p["lastLine"]))
    return 0


def cmd_watch(argv: list[str]) -> int:
    import time
    try:
        while True:
            p = runs.progress()
            if not p.get("run"):
                print(A.dim("no run to watch"))
                return 1
            line = f"\r  {A.green('run') if p['running'] else A.dim('stopped')}  " \
                   f"{p.get('roundsOk',0)}/{p.get('roundsTotal',0)} rounds  " \
                   f"{p.get('findings',0)} findings  {A.dim(str((p.get('lastLine') or ''))[:70])}"
            sys.stdout.write(line + " " * 10)
            sys.stdout.flush()
            if not p["running"]:
                print()
                return 0
            time.sleep(3)
    except KeyboardInterrupt:
        print()
        return 0


def cmd_targets(argv: list[str]) -> int:
    for n in config.targets():
        t = config.target_get(n) or {}
        problems = runs._validate(n)
        mark = A.green("● ") if n == config.active_target() else "  "
        tag = A.green("ready") if not problems else A.red(problems[0][:70])
        print(f"{mark}{n:<28} {tag}   {A.dim(str(t.get('dir',''))[:60])}")
    return 0


def cmd_run(argv: list[str]) -> int:
    target = argv[0] if argv else None
    if target and target not in config.targets():
        print(A.red(f"no target named {target}"), file=sys.stderr)
        print(A.dim("known: " + ", ".join(config.targets())), file=sys.stderr)
        return 2
    problems = runs._validate(target or config.active_target())
    if problems:
        for p in problems:
            print(A.red("  ✗ " + p), file=sys.stderr)
        return 2
    res = runs.start(target)
    if not res["ok"]:
        print(A.red("  ✗ " + str(res.get("reason"))), file=sys.stderr)
        return 1
    r = res["run"]
    print(A.green(f"✓ started  id {r['id']}  pid {r['pid']}"))
    print(A.dim(f"  log   {r['log']}"))
    print(A.dim(f"  watch orch watch   stop orch stop"))
    return 0


def cmd_stop(argv: list[str]) -> int:
    run = runs.current_run()
    if not run:
        print(A.dim("no run to stop"))
        return 1
    res = runs.stop(run["id"])
    print(A.green("✓ stopped") if res.get("stopped") else A.dim(str(res.get("reason"))))
    return 0


def cmd_findings(argv: list[str]) -> int:
    sev = None
    limit = 200
    i = 0
    while i < len(argv):
        if argv[i] == "--severity" and i + 1 < len(argv):
            sev = argv[i + 1].upper(); i += 2; continue
        if argv[i] == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1]); i += 2; continue
        i += 1
    fs = runs.findings(limit=limit)
    if sev:
        fs = [f for f in fs if str(f.get("severity") or f.get("criticality") or "").upper() == sev]
    if not fs:
        print(A.dim("no findings yet"))
        return 1
    for f in fs:
        s = str(f.get("severity") or f.get("criticality") or "?")
        col = {"CRITICAL": "red", "HIGH": "yellow", "MEDIUM": "cyan"}.get(s.upper(), "grey")
        title = str(f.get("title") or f.get("issue") or f.get("finding") or "")[:76]
        print(f"{A.style(s[:4].ljust(4), col)} {title}")
        print(A.dim(f"      {str(f.get('file') or '')[:84]}"))
    return 0


def cmd_config(argv: list[str]) -> int:
    if not argv:
        problem = config.config_problem()
        if problem:
            # Nothing below this line is the user's configuration — it is the defaults standing
            # in for a file that could not be read. Showing them without saying so is how a
            # "my settings reverted" report becomes an hour of looking at the wrong thing.
            print(A.red("  ⚠ the config file could not be read: " + problem["error"][:100]))
            print(A.dim(f"    {problem['file']}"))
            print(A.dim("    showing DEFAULTS, not your settings. Fix the file, or reset it:"))
            print(A.dim("      orch config repair      (backs the broken file up, writes a clean one)"))
            print()
        for r in config.resolved_all():
            val = "••••••" if r["kind"] == "secret" and r["value"] else (
                json.dumps(r["value"]) if isinstance(r["value"], list) else str(r["value"]))
            tag = "" if r["source"] == "file" else A.dim(f"  ({r['source']})")
            print(f"  {A.b(r['id']):<32} {val[:44]}{tag}")
        return 0
    action = argv[0]
    if action == "get" and len(argv) > 1:
        _out({"id": argv[1], "value": config.resolve(argv[1])[0], "source": config.resolve(argv[1])[1]})
        return 0
    if action in ("repair", "fix"):
        return cmd_config_repair()
    if action == "set" and len(argv) > 2:
        res = config.setting_save(argv[1], " ".join(argv[2:]))
        print(A.green(f"✓ {res['value']}") if res["ok"] else A.red("✗ " + str(res.get("reason"))))
        if res.get("note"):
            # A value can be legal and still mean something the value does not show. Saying so
            # at save time is the only moment the user is looking at that setting.
            print(A.yellow("  ⚠ " + res["note"]))
        if res.get("warning"):
            print(A.yellow("  ⚠ " + res["warning"]))
            print(A.dim("    the original is preserved at " + str(res.get("backedUp"))))
        return 0 if res["ok"] else 1
    if action == "reset" and len(argv) > 1:
        res = config.setting_reset(argv[1])
        print(A.green(f"✓ {argv[1]} -> {res.get('value')}") if res["ok"] else A.red("✗ " + str(res.get("reason"))))
        if res.get("warning"):
            print(A.yellow("  ⚠ " + res["warning"]))
            print(A.dim("    the original is preserved at " + str(res.get("backedUp"))))
        return 0 if res["ok"] else 1
    print(USAGE)
    return 2


def cmd_agents(argv: list[str]) -> int:
    from . import mcp_server as m
    res = m._agents_for(argv[0] if argv else None)
    if res.get("isError"):
        print(res["content"][0]["text"])
        return 1
    print(A.b(f"{res['count']} agents per batch  (label: {res['label']})"))
    for a in res["agents"]:
        kind = A.cyan("specialist") if a["kind"] == "specialist" else A.dim("generic")
        print(f"  {a['order']:>2}. w{a['weight']:<4} {a['name']:<36} {kind}")
    return 0


def cmd_models(argv: list[str]) -> int:
    agg = runs.aggregate_health()
    if agg["ok"]:
        print(A.green("endpoint up") + A.dim("  " + str(agg.get("url"))))
        for m in agg.get("models") or []:
            print("  " + m)
    else:
        print(A.red("endpoint down") + "  " + str(agg.get("error")))
    print()
    for l in runs.lane_health():
        h = l.get("health") or {}
        if l.get("ok"):
            b = A.green("browser") if h.get("browserAlive") else A.yellow("no browser")
            print(f"  {l['lane']:<4} :{l['port']}  {b}")
        else:
            print(f"  {l['lane']:<4} :{l['port']}  {A.red('unreachable')}")
    allowed = config.resolve("modelAllowlist")[0]
    print()
    print(f"  allowlist: {A.b(str(allowed))}")
    return 0


def cmd_doctor(argv: list[str]) -> int:
    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  {A.green('✓') if good else A.red('✗')} {label:<34} {A.dim(detail[:70])}")

    py = runs.find_python()
    can = False
    try:
        import subprocess
        can = subprocess.run([py, "-c", "import aiohttp"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        pass
    check("python can import aiohttp", can, py)
    eng = str(config.resolve("enginePy")[0])
    check("engine present", __import__("pathlib").Path(eng).exists(), eng)
    agg = runs.aggregate_health()
    check("model endpoint reachable", agg["ok"], str(agg.get("error") or agg.get("url")))
    lanes = runs.lane_health()
    live = [l["lane"] for l in lanes if l.get("ok")]
    check("at least one lane up", bool(live), ", ".join(live) or "none")
    for l in lanes:
        h = l.get("health") or {}
        if l.get("ok"):
            check(f"lane {l['lane']} browser attached", bool(h.get("browserAlive")), "")
    t = config.active_target()
    problems = runs._validate(t)
    check(f"active target '{t}' runnable", not problems, problems[0] if problems else "")
    out = str(config.resolve("outputDir")[0])
    check("output directory writable", __import__("pathlib").Path(out).exists()
          or __import__("os").access(str(__import__("pathlib").Path(out).parent), __import__("os").W_OK), out)
    print()
    print(A.green("ready to run") if ok else A.yellow("fix the ✗ lines above before running"))
    return 0 if ok else 1


def cmd_mcp(argv: list[str]) -> int:
    from . import mcp_server as m
    return m.main()


def cmd_ui(argv: list[str]) -> int:
    from . import screens
    if not A._tty_available():
        # The panel owns the terminal (raw mode, alternate screen). Refusing with one clear
        # line is far better than a termios traceback from inside the first draw.
        print("orch: the control panel needs a terminal. Use a subcommand instead, e.g.")
        print("      orch status | orch targets | orch config | orch findings")
        return 2
    try:
        with A.Screen():
            view = "dashboard"
            while True:
                if view == "dashboard":
                    view = screens.screen_dashboard() or "main"
                elif view == "main":
                    view = screens.screen_main()
                elif view == "run":
                    view = screens.screen_run()
                elif view == "targets":
                    view = screens.screen_targets()
                elif view == "config":
                    view = screens.screen_config()
                elif view == "models":
                    view = screens.screen_models()
                elif view == "findings":
                    view = screens.screen_findings()
                elif view == "agents":
                    view = screens.screen_agents()
                elif view == "mcp":
                    view = screens.screen_mcp()
                elif view == "scope":
                    view = screens.screen_scope()
                elif view == "batches":
                    view = screens.screen_batches()
                elif view == "prompt":
                    view = screens.screen_prompt()
                elif view == "run_control":
                    view = screens.screen_run_control()
                else:
                    view = "main"
    except A.QuitError:
        pass
    except KeyboardInterrupt:
        pass
    return 0



def cmd_scan(argv: list[str]) -> int:
    """What a run WOULD audit, before starting one."""
    r = runs.scan_target(argv[0] if argv else None)
    if r.get("error"):
        print(A.red("  ✗ " + str(r["error"])))
        return 1
    print(f"  {A.b(str(r['files']))} files  ({r['source']})")
    for m in r.get("missing", [])[:10]:
        print(A.red(f"  ✗ listed file does not exist: {m}"))
    for s in r.get("sample", [])[:8]:
        print(A.dim("    " + s))
    if r.get("roots"):
        for x in r["roots"]:
            print(A.dim("    root: " + x))
    return 0


def cmd_estimate(argv: list[str]) -> int:
    r = runs.estimate(argv[0] if argv else None)
    if r.get("error"):
        print(A.red("  ✗ " + str(r["error"])))
        return 1
    print(f"  files        {A.b(str(r['files']))}")
    print(f"  batches      {r['batches']}  ({r['batchSize']} files each)")
    print(f"  passes       {r['passes']}")
    print(f"  total rounds {r['totalRounds']}  ({r['roundsPerBatch']} per batch)")
    print(f"  avg round    {r['avgRoundSeconds']}s  [{A.dim(r['basis'])}]")
    print(f"  estimate     {A.b(str(r['estimatedHours']))} hours")
    print(A.dim("  " + r["note"]))
    return 0


def cmd_pause(argv: list[str]) -> int:
    r = runs.pause("")
    print(A.green(f"✓ paused pid {r.get('pid')}") if r.get("ok") else A.red("✗ " + str(r.get("reason"))))
    return 0 if r.get("ok") else 1


def cmd_resume(argv: list[str]) -> int:
    r = runs.resume("")
    print(A.green(f"✓ resumed pid {r.get('pid')}") if r.get("ok") else A.red("✗ " + str(r.get("reason"))))
    return 0 if r.get("ok") else 1


def cmd_batches(argv: list[str]) -> int:
    bs = runs.batch_files()
    if not bs:
        print(A.dim("  no batches saved yet (a batch saves only when all of its rounds finish)"))
        return 1
    for b in bs:
        mark = A.green("✓") if b["findings"] else A.yellow("·")
        print(f"  {mark} {b['pass']}/{b['batch']:<14} {b['findings']:>3} findings  {b['bytes']:>7}B")
        for f in b["files"][:3]:
            print(A.dim("      " + str(f)[:84]))
    return 0


def cmd_discard(argv: list[str]) -> int:
    if len(argv) < 2:
        print(A.red("usage: orch discard <pass_1> <batch_003>"), file=sys.stderr)
        return 2
    r = runs.delete_batch(None, argv[0], argv[1])
    if r.get("ok"):
        print(A.green(f"✓ discarded -> {r['discarded']}"))
        print(A.dim("  the next resumed run will re-audit it"))
        return 0
    print(A.red("✗ " + str(r.get("reason"))))
    return 1


def cmd_export(argv: list[str]) -> int:
    fmt = "json"
    for i, a in enumerate(argv):
        if a == "--format" and i + 1 < len(argv):
            fmt = argv[i + 1]
    r = runs.findings_export(fmt=fmt)
    if r.get("ok"):
        print(A.green(f"✓ {r['findings']} findings -> {r['path']}"))
        return 0
    print(A.red("✗ " + str(r.get("reason"))))
    return 1


def cmd_prompt(argv: list[str]) -> int:
    """The exact system prompt a persona receives. The only way to see what an auditor
    was actually told."""
    from . import mcp_server as m
    r = m._prompt_preview({"agent": argv[0] if argv else None})
    if isinstance(r, dict) and r.get("isError"):
        print(A.red(r["content"][0]["text"]))
        return 1
    print(A.b(f"{r['agent']}  (w{r['weight']}, {r['chars']} chars)"))
    print(A.dim(f"  domain context included: {r['domainContextIncluded']}"))
    print(A.hr() if hasattr(A, 'hr') else "-" * A.term_width())
    print(r["prompt"][:6000])
    return 0


def cmd_logs(argv: list[str]) -> int:
    for l in runs.logs_list():
        print(f"  {l['modified']}  {l['bytes']:>9}B  {l['name']}")
    return 0


def cmd_probe(argv: list[str]) -> int:
    r = runs.endpoint_probe()
    if r.get("ok"):
        print(A.green(f"✓ {r['model']} answered in {r['elapsedMs']}ms") + A.dim(f'  "{r["reply"]}"'))
        return 0
    # .get() rather than [], so a caller can never be broken by a shortened probe result —
    # that is exactly how this line produced a KeyError over a perfectly clear error message.
    who = r.get("model") or "the endpoint"
    ms = r.get("elapsedMs")
    head = f"✗ {who} failed" + (f" after {ms}ms" if ms else "")
    print(A.red(head) + A.dim("  " + str(r.get("error"))))
    return 1



def cmd_config_repair() -> int:
    """Recover from a config file that does not parse.

    The broken file is backed up, never deleted: it is the user's own text and the only record
    of the settings they had, so it must survive even if it is unusable.
    """
    problem = config.config_problem()
    file = config.CONFIG_FILE
    if not problem:
        print(A.green("✓ the config file reads fine — nothing to repair"))
        return 0
    print(A.red("  cannot read " + str(file)))
    print(A.dim("    " + problem["error"][:140]))
    print()
    if not A.confirm("  Back it up and write a clean default config in its place?", default=True):
        return 1
    import shutil
    stamp = __import__("time").strftime("%Y%m%d-%H%M%S")
    dest = file.with_name(file.name + f".bak-broken-{stamp}")
    try:
        if file.exists():
            shutil.copy2(file, dest)
    except Exception as e:
        print(A.red("  ✗ could not back the file up: " + str(e)))
        return 1
    data, _ = config.load_raw()          # defaults, since the file is unreadable
    config.save_raw(data, file)
    print(A.green("✓ clean config written"))
    print(A.dim("  your original is at " + str(dest)))
    print(A.dim("  copy any setting you want back out of it and set it with `orch config set`"))
    return 0

COMMANDS = {
    "run": cmd_run, "status": cmd_status, "watch": cmd_watch, "stop": cmd_stop,
    "targets": cmd_targets, "findings": cmd_findings, "config": cmd_config,
    "agents": cmd_agents, "models": cmd_models, "doctor": cmd_doctor,
    "mcp": cmd_mcp, "ui": cmd_ui,
    # added: the operations the panel and MCP expose
    "scan": cmd_scan, "estimate": cmd_estimate, "batches": cmd_batches,
    "pause": cmd_pause, "resume": cmd_resume, "discard": cmd_discard,
    "export": cmd_export, "prompt": cmd_prompt, "logs": cmd_logs, "probe": cmd_probe,
    "repair": cmd_config_repair,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return cmd_ui([])
    cmd = argv[0]
    rest = argv[1:]
    if cmd in ("help", "-h", "--help"):
        print(USAGE)
        return 0
    if cmd in ("version", "--version"):
        print(f"orch {config.ROOT.name}")
        return 0
    fn = COMMANDS.get(cmd)
    if not fn:
        print(A.red(f"unknown command: {cmd}"), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    return fn(rest)


if __name__ == "__main__":
    sys.exit(main())
