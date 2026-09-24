#!/usr/bin/env python3
"""aggregate.py — one OpenAI-compatible front for several lanes.

WHY THIS EXISTS
  The audit/execute engines are OpenAI-compatible HTTP clients: they read
  `GET {API_BASE}/models` to build a fallback chain and POST to
  `{API_BASE}/chat/completions`. Our lanes are not one server — they are
  separate webchat gateways (slow, per-account) plus OpenRouter (fast, free
  pool). Exposing them behind one endpoint is what lets the engine's existing
  per-model cooldown and ladder logic do its job, instead of teaching the engine
  about every lane.

MODEL IDS MAP TO LANES
  `ds`  -> the DeepSeek webchat gateway      (slow, anti-ban paced)
  `gm`  -> the Gemini webchat gateway        (slow, thinking)
  `cg`  -> the ChatGPT webchat gateway       (slow)
  `or-<openrouter-id>` -> OpenRouter itself  (fast, free tier)
  A REQUEST for a model this server does not recognise is 404, so a typo fails
  loudly instead of silently landing on the wrong lane.

FAILURE IS EXPLICIT. A lane that refuses returns its own error with the lane
name in the message. Returning an empty 200 would let the engine read a dead
lane as "the model chose not to answer" — the exact confusion that parks a
working lane and stalls every step.
"""
import asyncio
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

PORT = int(os.environ.get("AGG_PORT", "8090"))

LANES = {
    "ds": {
        "url": os.environ.get("AGG_DS_URL", "http://127.0.0.1:8081/v1/chat/completions"),
        "model": os.environ.get("AGG_DS_MODEL", "anymodel"),
        "timeout": int(os.environ.get("AGG_DS_TIMEOUT", "300")),
    },
    "gm": {
        "url": os.environ.get("AGG_GM_URL", "http://127.0.0.1:8085/v1/chat/completions"),
        "model": os.environ.get("AGG_GM_MODEL", "gemini-webchat"),
        "timeout": int(os.environ.get("AGG_GM_TIMEOUT", "360")),
    },
    "cg": {
        "url": os.environ.get("AGG_CG_URL", "http://127.0.0.1:8087/v1/chat/completions"),
        "model": os.environ.get("AGG_CG_MODEL", "chatgpt-webchat"),
        "timeout": int(os.environ.get("AGG_CG_TIMEOUT", "300")),
    },
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free, coding-capable ids. Measured live 2026-09-24: nemotron-3-ultra and
# nex-n2.5-pro both answer real content; poolside/laguna-s-2.1 was 429.
# A model that can never answer is worse than a missing one, so laguna is last.
OPENROUTER_FREE = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nex-agi/nex-n2.5-pro:free",
    "nex-agi/nex-n2.5-mini:free",
    "poolside/laguna-s-2.1:free",
]

# Ids the ENGINE asks for that no upstream understands. "auto/best-reasoning" is the
# engine's own PRIMARY_MODEL (OmniRoute routing syntax); unmapped it 404s on every call.
# Map it to the model that measurably does the work rather than editing the engine.
AUTO_ALIASES = {
    "auto/best-reasoning": "nvidia/nemotron-3-ultra-550b-a55b:free",
}


def openrouter_key():
    for p in (os.environ.get("OPENROUTER_KEY_FILE", ""),
              os.path.expanduser("~/.config/orch/openrouter.token"),
              os.path.expanduser("~/.claude/openrouter.token")):
        if not p:
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                k = fh.read().strip()
            if k:
                return k
        except OSError:
            continue
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def model_index():
    """id -> (lane, upstream model). The ids the engine sees."""
    idx = {}
    for name, lane in LANES.items():
        idx[name] = (name, lane["model"])
    for mid in OPENROUTER_FREE:
        idx["or-" + mid] = ("or", mid)
    # The engine's built-in PRIMARY_MODEL is "auto/best-reasoning" — OmniRoute routing
    # syntax that no upstream understands. Unmapped it 404s on every call and burns a
    # round each time, so point it at the model that does the real work. Keep the id
    # the engine knows as the alias rather than editing the engine's constant.
    for alias, mid in AUTO_ALIASES.items():
        idx[alias] = ("or", mid)
    return idx


def post_json(url, payload, headers, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers=headers, method="POST")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def _normalise_error_body(code, text, model_id):
    """Turn an error-shaped 200 into an honest HTTP status.

    OpenRouter answers HTTP 200 with ``{"error": {...}}`` for upstream failures
    (measured: "Upstream error from Nvidia: Service temporarily overloaded",
    "Provider returned error / 429"). Passing that through as a success makes the
    engine do ``data['choices'][0]`` on a body with no ``choices`` and raise
    KeyError('choices') — which reads exactly like a client bug and buries the real
    cause. A response with no ``choices`` is not a success; say so with a status.
    """
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return code, text
    if isinstance(obj, dict) and "choices" not in obj and "error" in obj:
        err = obj.get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err)
        upstream = err.get("code") if isinstance(err, dict) else None
        status = upstream if isinstance(upstream, int) and 400 <= upstream < 600 else 502
        return status, json.dumps({"error": {"message": f"aggregate: {model_id}: {msg}"}})
    return code, text


def call_lane(lane, model_id, payload):
    if lane == "or":
        key = openrouter_key()
        if not key:
            return 503, json.dumps({"error": {"message": "aggregate: no OpenRouter key"}})
        body = dict(payload)
        body["model"] = model_id
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
            "HTTP-Referer": "https://github.com/ronisaguey-ux/AI-coding-orchestrator",
            "X-Title": "AI-coding-orchestrator",
        }
        try:
            code, text = post_json(OPENROUTER_URL, body, headers, int(os.environ.get("AGG_OR_TIMEOUT", "120")))
            return _normalise_error_body(code, text, model_id)
        except HTTPError as e:
            return _normalise_error_body(e.code, e.read().decode("utf-8", "replace"), model_id)
        except (URLError, TimeoutError, OSError) as e:
            return 502, json.dumps({"error": {"message": f"aggregate: openrouter unreachable: {e}"}})

    spec = LANES[lane]
    body = dict(payload)
    # The webchat gateways accept their own webchat token, not an arbitrary id.
    body["model"] = spec["model"]
    headers = {"Content-Type": "application/json"}
    try:
        code, text = post_json(spec["url"], body, headers, spec["timeout"])
        out = json.loads(text)
        # Report the model the ENGINE asked for, so its per-model health tracking
        # keys on the id it knows rather than the gateway's placeholder.
        out.setdefault("model", payload.get("model"))
        return code, json.dumps(out)
    except HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        return e.code, json.dumps({"error": {"message": f"aggregate: lane {lane} failed: {detail[:400]}"}})
    except (URLError, TimeoutError, OSError) as e:
        return 502, json.dumps({"error": {"message": f"aggregate: lane {lane} unreachable: {e}"}})
    except json.JSONDecodeError:
        return 502, json.dumps({"error": {"message": f"aggregate: lane {lane} returned non-JSON"}})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[aggregate] " + (fmt % args) + "\n")

    def _send(self, code, payload):
        raw = payload.encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            # The engine abandons a call that outran its per-call budget. The client
            # is gone, so there is nowhere to write the answer: drop it quietly
            # instead of dumping a traceback over the log for every timed-out call.
            self.close_connection = True

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            data = [{"id": mid, "object": "model", "owned_by": idx[0]}
                    for mid, idx in sorted(model_index().items())]
            self._send(200, json.dumps({"object": "list", "data": data}))
            return
        self._send(404, json.dumps({"error": {"message": "aggregate: GET " + self.path}}))

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, json.dumps({"error": {"message": "aggregate: POST " + self.path}}))
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send(400, json.dumps({"error": {"message": f"aggregate: bad request JSON: {e}"}}))
            return
        mid = payload.get("model", "")
        idx = model_index()
        if mid not in idx:
            self._send(404, json.dumps({"error": {"message":
                f"aggregate: unknown model {mid!r}. Known: {', '.join(sorted(idx))}"}}))
            return
        lane, upstream = idx[mid]
        code, text = call_lane(lane, upstream, payload)
        # Log every request with its size and outcome. Without this a stalled audit is
        # indistinguishable from a hung lane: the engine prints nothing until a batch
        # completes, and the aggregate used to log only crashes — so "the audit is
        # stuck" had no evidence behind it. One line per call makes the bottleneck
        # visible (which lane, how big a prompt, what came back).
        try:
            prompt_chars = sum(len(str(m.get("content", ""))) for m in (payload.get("messages") or []))
            content = ""
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and parsed.get("choices"):
                    content = str(parsed["choices"][0].get("message", {}).get("content") or "")
                elif isinstance(parsed, dict) and parsed.get("error"):
                    content = "ERROR: " + str(parsed["error"])[:200]
            except (json.JSONDecodeError, TypeError, IndexError):
                content = "UNPARSEABLE"
            sys.stderr.write(
                f"[aggregate] {mid} <- {lane} | prompt {prompt_chars}c | http {code} | "
                f"reply {len(content)}c | {content[:120]!r}\n"
            )
        except Exception as e:  # logging must never break a response
            sys.stderr.write(f"[aggregate] log failed: {e}\n")
        self._send(code, text)


if __name__ == "__main__":
    print(f"[aggregate] listening on 127.0.0.1:{PORT}", flush=True)
    for n, l in LANES.items():
        print(f"[aggregate] lane {n:3s} -> {l['url']} (model {l['model']})", flush=True)
    print(f"[aggregate] lane or  -> openrouter free ({len(OPENROUTER_FREE)} models)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
