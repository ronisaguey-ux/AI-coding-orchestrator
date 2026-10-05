"""Regression tests for the engine/queue state + git paths (audit/engine-queue).

Each test pins a defect that was reproduced, not inferred. Nothing here touches
the live state file or the live repo: every test builds its own tmp_path repo or
state file.

  A1  orch_queue.drive_batch raised `NameError: name 'on_step_done' is not
      defined` on the first completed step - it called a hook the signature never
      accepted. 8 of the module's own tests failed on main.
  A2  execute.load_state() answered an UNREADABLE state file with {"steps": {}},
      which reads as "no step has ever been recorded". The engine then re-ran the
      plan from zero and save_state_serialized wrote that memory copy over the
      file (its own re-read of the same bytes also fails, so the merge is skipped)
      - destroying every result the file held.
  A3  run_group:verify_red_exhausted retired a step whose apply had SUCCEEDED and
      whose edit was verifiably on disk, without reading the file. Yellow means "a
      lane looked and could not"; the work had landed.
  A4  orch_git committed the WHOLE INDEX, so a file staged by any other process -
      including `webchat-api/.env`, which lives inside the repo - was swept into
      the step's commit and pushed to origin/main by _push_retry.
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

import execute
import orch_git
import orch_queue as oq


async def _noop_save(st):
    """Never touch a real state file from a test."""
    return True


def _async_noop(_tag):
    async def _f(*a, **k):
        return _tag
    return _f


# --------------------------------------------------------------- A1 ----------
def test_drive_batch_takes_the_step_done_hook_it_calls():
    """drive_batch must not NameError the first time a step finishes."""
    import inspect
    assert "on_step_done" in inspect.signature(oq.drive_batch).parameters


def test_drive_batch_completes_a_step_and_fires_the_hook():
    """End-to-end: two lanes, one step, hook fires, batch returns."""
    q = oq.BatchQueue(["S1"], {"S1": {"rounds": 0, "status": "pending"}},
                      steps_by_id={"S1": {"files": ["a.py"]}})
    roster = oq.LaneRoster(["L1"])
    done = []

    async def exec_step(lane, sid):
        return oq.StepOutcome("green", lane=lane)

    async def no_esc(sid):
        return "yellow"

    handoff = oq.StepHandoff()
    snap = asyncio.run(oq.drive_batch(
        q, roster, handoff, exec_step, no_esc, poll_s=0.005,
        on_step_done=lambda lane, sid, status: done.append((lane, sid, status))))

    assert snap == {"S1": "green"}
    assert done == [("L1", "S1", "green")], "on_step_done never fired"


# --------------------------------------------------------------- A2 ----------
def _isolate_state(tmp_path, monkeypatch):
    """Point execute at a private state file + repo. Returns the state path."""
    st = tmp_path / "exec_state.json"
    st.write_text(json.dumps({"steps": {}}))
    monkeypatch.setattr(execute, "STATE_FILE", st)
    return st


def test_load_state_absent_file_is_an_empty_plan(tmp_path, monkeypatch):
    """No file at all is legitimately empty - that must keep working."""
    monkeypatch.setattr(execute, "STATE_FILE", tmp_path / "nope.json")
    assert execute.load_state() == {"steps": {}}


def test_load_state_unreadable_file_must_not_read_as_an_empty_plan(tmp_path, monkeypatch):
    """THE DEFECT: a corrupt state file came back as {'steps': {}}.

    That is indistinguishable from "nothing has been recorded", so the engine
    re-ran the whole plan and then overwrote the file with the result.
    """
    st = _isolate_state(tmp_path, monkeypatch)
    st.write_text('{"steps": {"A": {"status": "green"')      # truncated write
    with pytest.raises(RuntimeError) as ei:
        execute.load_state()
    msg = str(ei.value)
    assert "refusing to start" in msg, msg
    # the unreadable bytes are kept so the run can be recovered
    assert any(p.name.endswith(".bak") for p in tmp_path.iterdir()), list(tmp_path.iterdir())


def test_load_state_non_object_json_must_not_read_as_an_empty_plan(tmp_path, monkeypatch):
    """A state file holding a JSON list is not an empty plan either."""
    st = _isolate_state(tmp_path, monkeypatch)
    st.write_text('[]')
    with pytest.raises(RuntimeError):
        execute.load_state()


def test_unreadable_state_cannot_be_overwritten_by_a_blank_run(tmp_path, monkeypatch):
    """The consequence, end to end.

    Before the fix, load_state() -> {"steps": {}} gave the engine an empty plan,
    and the first save_state_serialized() replaced the file with only the steps
    that run happened to touch — its own re-read of the same bytes also failed, so
    the merge was skipped and nothing was preserved. The engine's startup sequence
    must now abort at the load instead, leaving the bytes on disk untouched.
    """
    st = _isolate_state(tmp_path, monkeypatch)
    good = {"steps": {"A": {"rounds": 3, "status": "green"},
                      "B": {"rounds": 3, "status": "green"}}}
    st.write_text(json.dumps(good))
    original = st.read_text()
    st.write_text(original[:-5])                     # truncate: unreadable
    truncated = st.read_text()

    with pytest.raises(RuntimeError):
        execute.load_state()                          # startup aborts here ...

    assert st.read_text() == truncated, "the unreadable state was modified"
    assert json.loads(original) == good, "sanity: the good state really had records"


# --------------------------------------------------------------- A3 ----------
def _landed_repo(tmp_path):
    """A one-file repo whose content is already the step's new_string."""
    f = tmp_path / "mod.py"
    f.write_text("VALUE = 2\n")
    return f


def test_applied_edits_landed_reads_the_file(tmp_path, monkeypatch):
    """The evidence primitive the retire sites are supposed to consult."""
    monkeypatch.setattr(execute, "REPO", tmp_path)
    _landed_repo(tmp_path)                       # content is already the new_string
    rec = {"last_apply": {"ok": True, "edits": [
        {"file": "mod.py", "old_string": "VALUE = 1",
         "new_string": "VALUE = 2"}]}}
    assert execute._applied_edits_landed(rec) is True
    # and it is not satisfied by a claim: change the file back
    _landed_repo(tmp_path).write_text("VALUE = 1\n")
    assert execute._applied_edits_landed(rec) is False


class _Res:
    """Minimal stand-in for orch_lanes.LaneResult."""

    def __init__(self, content, ok=True, lane="fake", error=None):
        self.content, self.ok, self.lane, self.error = content, ok, lane, error
        self.model = "fake"

    @property
    def empty_answer(self):
        return self.ok and not self.content.strip()


def _drive_group_verify_red(tmp_path, monkeypatch, land_the_edit):
    """Run the GROUP path to the verify-red branch and return the step record.

    Two steps (so run_group does not delegate to run_step), rounds seeded one
    below the cap so the run lands exactly on MAX_ROUNDS, a lane that proposes a
    real edit for each step, and a verify lane that answers red with no
    corrections. `land_the_edit` decides whether the proposed edit is actually
    written to the file.
    """
    monkeypatch.setattr(execute, "REPO", tmp_path)
    (tmp_path / "a.py").write_text("A_OLD = 1\n")
    (tmp_path / "b.py").write_text("B_OLD = 1\n")

    monkeypatch.setattr(execute, "run_checks", lambda files: "")
    monkeypatch.setattr(execute, "save_state_serialized", _noop_save)
    monkeypatch.setattr(execute, "in_repo", lambda p: True)
    monkeypatch.setattr(execute, "file_text", lambda f, size_cap=24000: "")
    monkeypatch.setattr(execute.orch_git, "git_commit_step",
                        _async_noop("green,commit"))

    proposed = [
        {"step": "1", "file": "a.py", "old_string": "A_OLD = 1", "new_string": "A_NEW = 1"},
        {"step": "2", "file": "b.py", "old_string": "B_OLD = 1", "new_string": "B_NEW = 1"},
    ]

    async def repair(session, system, user, max_retries=1):
        return _Res(json.dumps({"edits": [dict(e) for e in proposed]}))

    async def verify(session, system, user):
        return _Res(json.dumps({"verdict": "red", "reason": "the check did not pass"}))

    real_apply = execute.apply_edits

    def apply(edits, default_file=""):
        ok, msg = real_apply(edits, default_file=default_file)
        if land_the_edit:
            # the content really is on disk; a second apply cannot find old_string
            # again, which is exactly the live shape of this branch.
            return ok, msg
        for e in edits:                       # pretend nothing reached the disk
            (tmp_path / str(e.get("file"))).write_text("A_OLD = 1\nB_OLD = 1\n")
        return ok, msg

    monkeypatch.setattr(execute, "repair_with_retry", repair)
    monkeypatch.setattr(execute, "lane_call_result", verify)
    monkeypatch.setattr(execute, "apply_edits", apply)

    steps = [{"finding_id": "G1", "title": "t1", "files": ["a.py"]},
             {"finding_id": "G2", "title": "t2", "files": ["b.py"]}]
    st = {"steps": {"G1": {"rounds": execute.MAX_ROUNDS - 1, "status": "pending"},
                    "G2": {"rounds": execute.MAX_ROUNDS - 1, "status": "pending"}}}

    asyncio.run(execute.run_group(object(), steps, st))
    return st


def test_verify_red_exhausted_greens_landed_work_instead_of_retiring_it(
        tmp_path, monkeypatch):
    """THE DEFECT: the group retire site never read the file it was retiring.

    The apply SUCCEEDED and the edit is in the file; only the verify lane said
    red. The site must read the artifact and record green, exactly as
    run_step:rounds_exhausted does — otherwise a step whose work landed is filed
    as a failure and the plan re-does the work later.
    """
    st = _drive_group_verify_red(tmp_path, monkeypatch, land_the_edit=True)
    rec = st["steps"]["G1"]
    assert rec["status"] == "green", (
        f"landed work was retired as {rec['status']}: "
        f"{rec.get('yellow_justification') or rec.get('escalated_reason')}")
    assert "on disk" in str(rec.get("resolved_by") or "")
    assert rec["last_apply"]["verified_by"] == \
        "run_group:verify_red_exhausted:content-on-disk"


def test_verify_red_exhausted_still_retires_work_that_never_landed(
        tmp_path, monkeypatch):
    """The guard must not invent greens: nothing on disk -> still retired."""
    st = _drive_group_verify_red(tmp_path, monkeypatch, land_the_edit=False)
    rec = st["steps"]["G1"]
    assert rec["status"] in ("yellow", "escalated"), rec["status"]
    assert "verified_by" not in (rec.get("last_apply") or {})


# --------------------------------------------------------------- A4 ----------
def _scratch_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for cmd in (["git", "init", "-q", "."],
                ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"]):
        subprocess.run(cmd, cwd=path, check=True, capture_output=True)
    return path


def _git(repo: Path, *args) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    return r.stdout


def test_commit_step_never_sweeps_a_foreign_staged_file(tmp_path, monkeypatch):
    """THE DEFECT: `git commit -m msg` committed the whole index.

    A `.env` staged by another process ended up inside this step's commit, and
    _push_retry then pushed it to origin/main.
    """
    repo = _scratch_repo(tmp_path / "repo")
    (repo / ".env").write_text("BASE\n")
    (repo / "stepfile.py").write_text("BASE\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    # another process stages a credentials file ...
    (repo / ".env").write_text("SECRET=leaked\n")
    _git(repo, "add", ".env")
    # ... and the engine stages its own step file
    (repo / "stepfile.py").write_text("FIXED\n")

    monkeypatch.setattr(orch_git, "REPO", repo)
    monkeypatch.setattr(orch_git, "_assert_oculus_repo", lambda: None)
    monkeypatch.setattr(orch_git, "_remote_with_token", lambda: "")
    monkeypatch.setattr(orch_git, "_push_retry", lambda attempt=0: None)

    rec = {"last_apply": {"ok": True, "edits": [{"file": "stepfile.py"}]}}
    orch_git._commit_blocking("S#1", ["stepfile.py"])

    changed = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
    assert changed.split() == ["stepfile.py"], (
        f"the step commit also carried {changed.split()}")
    # the secret is not in the commit ...
    assert _git(repo, "show", "HEAD:.env").strip() == "BASE", "credential leaked"
    # ... and the other process's staged file is still theirs to commit
    assert ".env" in _git(repo, "diff", "--cached", "--name-only")


def test_commit_step_still_commits_its_own_new_file(tmp_path, monkeypatch):
    """A pathspec commit must not break the create-a-file case."""
    repo = _scratch_repo(tmp_path / "repo2")
    (repo / "keep.py").write_text("k\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "fresh.py").write_text("new module\n")

    monkeypatch.setattr(orch_git, "REPO", repo)
    monkeypatch.setattr(orch_git, "_assert_oculus_repo", lambda: None)
    monkeypatch.setattr(orch_git, "_remote_with_token", lambda: "")
    monkeypatch.setattr(orch_git, "_push_retry", lambda attempt=0: None)

    orch_git._commit_blocking("S#2", ["fresh.py"])
    assert "fresh.py" in _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")


def test_commit_step_with_nothing_of_its_own_to_do_is_quiet(tmp_path, monkeypatch):
    """Foreign staged content must not make an unchanged step take the commit path."""
    repo = _scratch_repo(tmp_path / "repo3")
    (repo / ".env").write_text("BASE\n")
    (repo / "stepfile.py").write_text("SAME\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / ".env").write_text("SECRET=leaked\n")
    _git(repo, "add", ".env")

    monkeypatch.setattr(orch_git, "REPO", repo)
    monkeypatch.setattr(orch_git, "_assert_oculus_repo", lambda: None)
    monkeypatch.setattr(orch_git, "_remote_with_token", lambda: "")
    monkeypatch.setattr(orch_git, "_push_retry", lambda attempt=0: None)

    before = _git(repo, "rev-parse", "HEAD")
    orch_git._commit_blocking("S#3", ["stepfile.py"])
    assert _git(repo, "rev-parse", "HEAD") == before, (
        "committed with nothing of its own staged, on the strength of a "
        "foreign staged file")


def test_green_truth_rescue_commit_never_sweeps_a_foreign_staged_file(
        tmp_path, monkeypatch):
    """Same defect in the watch's rescue path (execute.py runs it at startup).

    green_truth_watch.rescue_commit did `git add -- files` then a bare
    `git commit -m msg`, so it committed the whole index and pushed it to
    origin/main.
    """
    import green_truth_watch as gtw

    repo = _scratch_repo(tmp_path / "repo4")
    (repo / ".env").write_text("BASE\n")
    (repo / "stepfile.py").write_text("BASE\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / ".env").write_text("SECRET=leaked\n")
    _git(repo, "add", ".env")
    (repo / "stepfile.py").write_text("FIXED\n")

    monkeypatch.setattr(gtw, "REPO", repo)
    # the push is the last statement; keep it off the network
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", _no_push(real_run))

    assert gtw.rescue_commit("S#9", [{"file": "stepfile.py"}]) is True
    changed = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
    assert changed.split() == ["stepfile.py"], (
        f"the rescue commit also carried {changed.split()}")
    assert _git(repo, "show", "HEAD:.env").strip() == "BASE", "credential leaked"
    assert ".env" in _git(repo, "diff", "--cached", "--name-only")


def _no_push(real_run):
    def run(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and "push" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return real_run(cmd, *a, **k)
    return run
