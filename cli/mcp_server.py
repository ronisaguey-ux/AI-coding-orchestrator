"""mcp_server.py — the orchestrator as an MCP server over stdio.

Any MCP-capable agent (opencode, Claude Code, an IDE, another orchestrator) can then drive a
full audit: configure targets and settings, launch a run, watch it, read the findings, verify
them against the code, and build an execution plan — the same surface the panel offers a human.

Zero dependencies, deliberately. An MCP server is started fresh by the client for every
session, so anything it needs must already be present; a package install would make the client
fail to connect with an error that looks like a protocol problem.

★ THE WIRE IS STDOUT. Every diagnostic goes to stderr. One stray print() corrupts the stream
and the client reports a JSON parse error, which reads as "the server is broken" rather than
"a function logged". The redirect below is installed before any module that might print.
"""

from __future__ import annotations

import sys

_REAL_STDOUT = sys.stdout


def _as_stderr(*a, **k):
    print(*a, file=sys.stderr, **k)
    sys.stderr.flush()


for _name in ("print", "log", "info", "warn", "debug", "error"):
    if hasattr(sys.stdout, _name):
        try:
            setattr(sys.stdout, _name, _as_stderr)
        except Exception:
            pass

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

from . import config, runs  # noqa: E402

PROTOCOL = "2024-11-05"
SERVER_NAME = "orch"
SERVER_VERSION = "1.0.0"

S = lambda d: {"type": "string", "description": d}
N = lambda d: {"type": "number", "description": d}
B = lambda d: {"type": "boolean", "description": d}
L = lambda d: {"type": "array", "description": d, "items": {"type": "string"}}


def _tool(name, description, props=None, required=None, handler=None):
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": props or {},
                            **({"required": required} if required else {})},
            "handler": handler}


def _text(s):
    return {"content": [{"type": "text", "text": s if isinstance(s, str) else json.dumps(s, indent=2)}]}


def _err(s):
    return {"content": [{"type": "text", "text": s if isinstance(s, str) else json.dumps(s, indent=2)}],
            "isError": True}


def _tools_module(name: str):
    """A helper under tools/ (verify_findings, build_plan, cross_eval), or None."""
    p = config.ROOT / "tools" / name
    return p if p.exists() else None


def _run_helper(script: str, args: list[str], timeout: int = 600) -> dict:
    p = _tools_module(script)
    if not p:
        return {"ok": False, "reason": f"tools/{script} is not present in this checkout"}
    try:
        r = subprocess.run([runs.find_python(), str(p)] + args,
                           cwd=str(config.ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"tools/{script} exceeded {timeout}s"}
    return {"ok": r.returncode == 0, "exit": r.returncode,
            "stdout": (r.stdout or "")[-8000:], "stderr": (r.stderr or "")[-3000:]}


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [

    # ── overview ─────────────────────────────────────────────────────────────
    _tool("orch_status",
          "One call for the whole picture: the active target, whether a run is going, its progress "
          "and finding count, the model endpoint, and each webchat lane's health. Start here.",
          None, None,
          lambda a: {"activeTarget": config.active_target(),
                     "targets": list(config.targets()),
                     "endpoint": runs.aggregate_health(),
                     "run": runs.progress(),
                     "batchesOnDisk": runs.batches_on_disk(),
                     "lanes": runs.lane_health()}),

    _tool("orch_health",
          "Health of the model endpoint and every webchat lane, plus which models the allowlist "
          "would accept. Use it when a run is slow or returning nothing.",
          None, None,
          lambda a: {"endpoint": runs.aggregate_health(),
                     "lanes": runs.lane_health(),
                     "allowlist": config.resolve("modelAllowlist")[0]}),

    # ── targets ──────────────────────────────────────────────────────────────
    _tool("orch_target_list",
          "Every configured audit target with its readiness: a target with a missing repo root or "
          "a listed file that does not exist cannot be run, and this says which.",
          None, None,
          lambda a: [{"name": n, "active": n == config.active_target(),
                      "problems": runs._validate(n), **(config.target_get(n) or {})}
                     for n in config.targets()]),

    _tool("orch_target_get",
          "One target in full, including the exact paths the engine will be given.",
          {"name": S("Target name.")}, ["name"],
          lambda a: config.target_get(a["name"]) or _err(f'no target named {a["name"]}')),

    _tool("orch_target_add",
          "Register a repo as an auditable target. `dir` is the repository root the file walk "
          "starts from. `label` selects the domain profile and the specialist personas — use "
          "helpotron, t2b or webchat-to-api-harness to get that target's specialists, or any "
          "other name for the generic set.",
          {"name": S("Target name (used on the command line and in this API)."),
           "dir": S("Repository root."),
           "label": S("Domain profile key. Defaults to the name."),
           "readme": S("README path, for documented-feature comparison."),
           "sot": S("Source-of-truth document. Optional."),
           "graph": S("Graphify graph.json. Optional."),
           "include_files": S("Explicit comma-separated file list. Set this to skip the directory walk entirely."),
           "extra_roots": S("Extra comma-separated roots to walk."),
           "task": S("Audit-protocol prompt file. Optional.")},
          ["name", "dir"],
          lambda a: config.target_save(a["name"], {
              "dir": a.get("dir"), "label": a.get("label") or a["name"],
              "readme": a.get("readme", ""), "sot": a.get("sot", ""),
              "graph": a.get("graph", ""), "includeFiles": a.get("include_files", ""),
              "extraRoots": a.get("extra_roots", ""), "task": a.get("task", "")})),

    _tool("orch_target_edit",
          "Change one field of an existing target.",
          {"name": S("Target name."), "field": S("One of: dir, label, readme, sot, graph, "
                                                 "includeFiles, extraRoots, task"),
           "value": S("New value.")}, ["name", "field", "value"],
          lambda a: config.target_save(a["name"], {a["field"]: a["value"]})),

    _tool("orch_target_remove",
          "Remove a target. Refused if it is the only one.",
          {"name": S("Target name.")}, ["name"],
          lambda a: config.target_remove(a["name"])),

    _tool("orch_target_activate",
          "Set the target that a plain `orch_run_start` uses.",
          {"name": S("Target name.")}, ["name"],
          lambda a: config.set_active_target(a["name"])),

    # ── configuration ────────────────────────────────────────────────────────
    _tool("orch_config_list",
          "Every setting with its effective value and where that value came from (file, environment, "
          "or default). Check `source` when a change appears to have no effect: an environment value "
          "wins over the file.",
          None, None,
          lambda a: [{"id": r["id"], "group": r["group"], "value": r["value"], "source": r["source"],
                      "kind": r["kind"], "help": r["help"]} for r in config.resolved_all()]),

    _tool("orch_config_get",
          "One setting's effective value, its origin, and its full description.",
          {"id": S("Setting id, e.g. batchSize or outputDir.")}, ["id"],
          lambda a: (lambda v: {"id": a["id"], "value": v[0], "source": v[1],
                                "spec": config.BY_ID.get(a["id"])})(
              config.resolve(a["id"])) if a["id"] in config.BY_ID
          else _err(f'unknown setting "{a["id"]}"')),

    _tool("orch_config_set",
          "Change one setting. Values are coerced to the setting's type and range-checked, so a bad "
          "value is refused with the reason rather than stored.",
          {"id": S("Setting id."), "value": S("New value. A list setting takes a comma-separated string.")},
          ["id", "value"],
          lambda a: config.setting_save(a["id"], a["value"])),

    _tool("orch_config_reset",
          "Return one setting to its default, and report whether an environment variable was also "
          "cleared (an env value would otherwise keep overriding the reset).",
          {"id": S("Setting id.")}, ["id"],
          lambda a: config.setting_reset(a["id"])),

    # ── runs ─────────────────────────────────────────────────────────────────
    _tool("orch_run_start",
          "Launch an audit as a DETACHED process and return its id immediately — it outlives this "
          "call and this session. Poll with orch_run_progress. Refused if a run is already going, "
          "or if the target has a missing path.",
          {"target": S("Target name. Defaults to the active one."),
           "passes": N("Override the number of passes."),
           "batch_size": N("Override files per batch. Lower this for a slow lane."),
           "concurrency": N("Override concurrent agent calls."),
           "resume": B("Skip batches already saved. Defaults to the configured value."),
           "output_dir": S("Override the output directory."),
           "models": L("Override the model allowlist, e.g. [\"ds\"]. First available wins each round.")},
          None,
          lambda a: runs.start(a.get("target"), {
              "passes": a.get("passes"), "batchSize": a.get("batch_size"),
              "concurrency": a.get("concurrency"), "outputDir": a.get("output_dir"),
              "modelAllowlist": a.get("models")}, resume=a.get("resume"))),

    _tool("orch_run_list",
          "Every run recorded, newest first, with whether its process is still alive.",
          None, None, lambda a: runs.list_runs()),

    _tool("orch_run_status",
          "The current (or a named) run: pid, alive, target, when it started, and its arguments.",
          {"run_id": S("Run id. Defaults to the current run.")}, None,
          lambda a: (runs.get(a["run_id"]) if a.get("run_id") else runs.current_run()) or {"run": None}),

    _tool("orch_run_progress",
          "Parsed progress of the current run: the batch being worked, each specialist round with its "
          "agent name, verdict, duration and finding count, plus the tail of the log. The engine does "
          "not expose state, so this reads its log.",
          {"run_id": S("Run id. Defaults to the current run.")}, None,
          lambda a: runs.progress(runs.get(a["run_id"]) if a.get("run_id") else None)),

    _tool("orch_run_log",
          "The last N lines of a run's log, raw. Use it when progress looks wrong — a failure records "
          "its reason here.",
          {"run_id": S("Run id. Defaults to the current run."),
           "lines": N("How many trailing lines. Default 120.")}, None,
          lambda a: (lambda r: {"log": r.get("log"),
                                "tail": (Path(r["log"]).read_text(errors="replace").splitlines()[-int(a.get("lines") or 120):]
                                         if r and Path(r.get("log") or "").exists() else [])})(
              runs.get(a["run_id"]) if a.get("run_id") else runs.current_run())),

    _tool("orch_run_stop",
          "Stop a run: SIGTERM to its process group, then SIGKILL if it does not exit.",
          {"run_id": S("Run id. Defaults to the current run.")}, None,
          lambda a: (lambda r: runs.stop(r["id"]) if r else {"ok": False, "reason": "no run to stop"})(
              runs.get(a["run_id"]) if a.get("run_id") else runs.current_run())),

    # ── results ──────────────────────────────────────────────────────────────
    _tool("orch_batches_list",
          "What is saved on disk per pass: how many batch files, how many findings. This is the "
          "durable record — it survives a stopped run and is what a resume builds on.",
          {"run_id": S("Run id. Defaults to the current run.")}, None,
          lambda a: runs.batches_on_disk(runs.get(a["run_id"]) if a.get("run_id") else None)),

    _tool("orch_findings_list",
          "Every finding written for the current run, flattened, with its file, severity and detail.",
          {"run_id": S("Run id. Defaults to the current run."),
           "limit": N("Maximum findings to return. Default 200."),
           "severity": S("Only this severity, e.g. CRITICAL.")}, None,
          lambda a: _filter_findings(a)),

    _tool("orch_findings_summary",
          "Findings counted by severity, by file, and by the round that produced them — the shape "
          "of what the run found, without the text.",
          {"run_id": S("Run id. Defaults to the current run.")}, None,
          lambda a: _findings_summary(a)),

    # ── verification / planning (the tools/ helpers) ──────────────────────────
    _tool("orch_verify_findings",
          "Run the verifier over a run's findings: it resolves each cited file and line range and "
          "decides SUPPORTED / REFUTED / UNVERIFIED from the code itself. A model cannot be trusted "
          "on its own findings, so this is the step that makes an audit usable. Writes "
          "verified_findings.json next to the run output.",
          {"findings": S("Path to a findings json. Defaults to the run's own output."),
           "target_dir": S("Repository root the citations are relative to.")}, None,
          lambda a: _run_helper("verify_findings.py", _verify_args(a))),

    _tool("orch_build_plan",
          "Turn verified findings into an executable plan: one step per change, each with the file, "
          "the exact change, the command that proves it, and what 'done' means.",
          {"verified": S("Path to verified_findings.json. Defaults to the run's output."),
           "out": S("Where to write the plan.")}, None,
          lambda a: _run_helper("build_plan.py", _plan_args(a))),

    _tool("orch_cross_eval",
          "Cross-evaluate a plan: check its steps against the code and say which are real, which are "
          "already fixed, and which contradict each other.",
          {"plan": S("Path to an execution plan json.")}, None,
          lambda a: _run_helper("cross_eval.py", _cross_args(a))),

    # ── personas ─────────────────────────────────────────────────────────────
    _tool("orch_agents_list",
          "The exact personas that will run for a target, in order, marked specialist or generic. "
          "A target with a domain profile swaps its weakest generic specialists for its own.",
          {"target": S("Target name. Defaults to the active one.")}, None,
          lambda a: _agents_for(a.get("target"))),

    _tool("orch_profile_get",
          "The domain context injected into every persona for a target: what the system is, its real "
          "surface, where its risk lives, and what is explicitly NOT a risk (which is what stops "
          "invented findings for the wrong kind of system).",
          {"target": S("Target name. Defaults to the active one.")}, None,
          lambda a: _profile_for(a.get("target"))),

    # ── files ────────────────────────────────────────────────────────────────
    _tool("orch_paths",
          "Every path this orchestrator resolves: config file, engine, output dir, run records, and "
          "the tools/ directory. Run this when a setting seems not to apply — it shows which file is "
          "in charge.",
          None, None,
          lambda a: {"configFile": str(config.CONFIG_FILE),
                     "repo": str(config.ROOT),
                     "engine": str(config.resolve("enginePy")[0]),
                     "python": runs.find_python(),
                     "outputDir": str(config.resolve("outputDir")[0]),
                     "runsDir": str(runs.runs_dir()),
                     "tools": sorted(p.name for p in (config.ROOT / "tools").glob("*.py"))
                              if (config.ROOT / "tools").exists() else []}),
]


# ── handler helpers ──────────────────────────────────────────────────────────

def _filter_findings(a):
    fs = runs.findings(runs.get(a["run_id"]) if a.get("run_id") else None,
                       limit=int(a.get("limit") or 200))
    sev = (a.get("severity") or "").upper()
    if sev:
        fs = [f for f in fs if str(f.get("severity") or f.get("criticality") or "").upper() == sev]
    return fs


def _findings_summary(a):
    fs = runs.findings(runs.get("run_id") if a.get("run_id") else None, limit=5000)
    by_sev, by_file = {}, {}
    for f in fs:
        s = str(f.get("severity") or f.get("criticality") or "UNKNOWN").upper()
        by_sev[s] = by_sev.get(s, 0) + 1
        fp = f.get("file") or "(no file)"
        by_file[fp] = by_file.get(fp, 0) + 1
    top = sorted(by_file.items(), key=lambda kv: -kv[1])[:25]
    return {"total": len(fs), "bySeverity": by_sev,
            "topFiles": [{"file": k, "findings": v} for k, v in top]}


def _run_out_dir(a) -> Path:
    run = runs.get(a.get("run_id")) if a.get("run_id") else runs.current_run()
    if run:
        return Path(run.get("outputDir") or "")
    return Path(str(config.resolve("outputDir")[0]))


def _verify_args(a):
    args = []
    out = _run_out_dir(a)
    if a.get("findings"):
        args += ["--findings", a["findings"]]
    elif out.exists():
        args += ["--findings", str(out)]
    if a.get("target_dir"):
        args += ["--target", a["target_dir"]]
    elif config.active_target():
        t = config.target_get(config.active_target()) or {}
        if t.get("dir"):
            args += ["--target", str(t["dir"])]
    return args


def _plan_args(a):
    args = []
    if a.get("verified"):
        args += ["--verified", a["verified"]]
    if a.get("out"):
        args += ["--out", a["out"]]
    return args


def _cross_args(a):
    return ["--plan", a["plan"]] if a.get("plan") else []


def _load_engine_lists(target: str | None):
    """Read AGENTS / DOMAIN_* out of the engine without importing it (it needs aiohttp)."""
    import ast as _ast

    engine = Path(str(config.resolve("enginePy")[0]))
    ns: dict = {"os": os}
    tree = _ast.parse(engine.read_text())
    keep = [n for n in tree.body
            if isinstance(n, (_ast.Assign, _ast.FunctionDef))
            and (getattr(n, "name", None) in ("AGENTS", "DOMAIN_PROFILES", "DOMAIN_AGENTS", "active_agents")
                 or (isinstance(n, _ast.Assign) and isinstance(n.targets[0], _ast.Name)
                     and n.targets[0].id in ("AGENTS", "DOMAIN_PROFILES", "DOMAIN_AGENTS")))]
    exec(compile(_ast.Module(body=keep, type_ignores=[]), "<engine>", "exec"), ns)
    tname = target or config.active_target()
    label = str((config.target_get(tname) or {}).get("label") or tname)
    ns["TARGET_LABEL"] = label
    return ns, label


def _agents_for(target: str | None):
    try:
        ns, label = _load_engine_lists(target)
    except Exception as e:
        return _err(f"could not read the engine personas: {e}")
    generic = {x["name"] for x in ns["AGENTS"]}
    agents = ns["active_agents"]()
    return {"target": target or config.active_target(), "label": label,
            "count": len(agents),
            "agents": [{"order": i + 1, "name": x["name"], "weight": x["weight"],
                        "kind": "generic" if x["name"] in generic else "specialist",
                        "focus": x.get("graph_focus")} for i, x in enumerate(agents)]}


def _profile_for(target: str | None):
    try:
        ns, label = _load_engine_lists(target)
    except Exception as e:
        return _err(f"could not read the engine profiles: {e}")
    prof = ns["DOMAIN_PROFILES"].get(label)
    return {"target": target or config.active_target(), "label": label,
            "profile": prof or None,
            "note": None if prof else ("no domain profile for this label - "
                                       "the auditor gets the generic personas")}


# ── JSON-RPC ─────────────────────────────────────────────────────────────────

TOOL_MAP = {t["name"]: t for t in TOOLS}


def dispatch(req: dict):
    """Handle one JSON-RPC request. Returns a response dict, or None for a notification."""
    method = req.get("method")
    rid = req.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}}}

    if method in ("notifications/initialized", "initialized"):
        return None

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "tools": [{k: v for k, v in t.items() if k != "handler"} for t in TOOLS]}}

    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = TOOL_MAP.get(name)
        if not tool:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"unknown tool: {name}"}}
        try:
            res = tool["handler"](args)
        except Exception as e:
            res = _err(f"{name} failed: {type(e).__name__}: {e}")
        if not isinstance(res, dict) or "content" not in res:
            res = _text(res)
        return {"jsonrpc": "2.0", "id": rid, "result": res}

    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}

    if rid is None:
        return None
    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _as_stderr(f"orch-mcp: unparseable request: {e}")
            continue
        try:
            resp = dispatch(req)
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": req.get("id"),
                    "error": {"code": -32603, "message": f"internal error: {e}"}}
        if resp is not None:
            _REAL_STDOUT.write(json.dumps(resp) + "\n")
            _REAL_STDOUT.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
