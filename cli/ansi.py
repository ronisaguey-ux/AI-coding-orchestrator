"""ansi.py — zero-dependency terminal primitives for the orch control panel.

Why no library: the panel has to run wherever the orchestrator runs (this box, a scratch
venv, a container) with no install step and no build step. Ink needs a transpile, curses
owns the terminal and fights a pipe. Escape codes and one raw-mode reader do everything
this panel needs.

Two rules learned the hard way in the harness CLI, both encoded here:
  * width() measures VISIBLE width.  A bold label measured with String.length is shorter
    than it renders, so a box border lands in the wrong column.
  * read_key() owns its own timer.  Racing the read from outside leaves the stdin listener
    registered, and the next call stacks a second one — one keypress then resolves twice.
"""

from __future__ import annotations

import os
import re
import select
import shutil
import sys
import termios
import time
import tty

# ── style ────────────────────────────────────────────────────────────────────

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"
UNDERLINE = "\x1b[4m"
REVERSE = "\x1b[7m"

FG = {
    "black": 30, "red": 31, "green": 32, "yellow": 33, "blue": 34,
    "magenta": 35, "cyan": 36, "white": 37, "grey": 90, "gray": 90,
    "brightred": 91, "brightgreen": 92, "brightyellow": 93, "brightblue": 94,
    "brightmagenta": 95, "brightcyan": 96, "brightwhite": 97,
}
BG = {k: v + 10 for k, v in FG.items()}

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def _wrap(code: str, s: str) -> str:
    return s if _NO_COLOR else code + s + RESET


def style(s: str, *specs: str) -> str:
    """Apply styles to a string. Unknown specs are ignored rather than raising."""
    if _NO_COLOR:
        return s
    pre = ""
    for spec in specs:
        if not spec:
            continue
        if spec in FG:
            pre += f"\x1b[{FG[spec]}m"
        elif spec.startswith("bg:"):
            name = spec[3:]
            if name in BG:
                pre += f"\x1b[{BG[name]}m"
        elif spec == "bold":
            pre += BOLD
        elif spec == "dim":
            pre += DIM
        elif spec == "italic":
            pre += ITALIC
        elif spec == "underline":
            pre += UNDERLINE
        elif spec == "reverse":
            pre += REVERSE
    return pre + s + RESET if pre else s


b = lambda s: style(s, "bold")
dim = lambda s: style(s, "dim")
red = lambda s: style(s, "red")
green = lambda s: style(s, "green")
yellow = lambda s: style(s, "yellow")
cyan = lambda s: style(s, "cyan")
blue = lambda s: style(s, "blue")
magenta = lambda s: style(s, "magenta")
grey = lambda s: style(s, "grey")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def visible_width(s: str) -> int:
    """The number of terminal columns a string occupies.

    ANSI escapes take no columns; a wide CJK character takes two. Getting this wrong is
    what makes a border ragged, so it is measured rather than assumed.
    """
    plain = ANSI_RE.sub("", s)
    import unicodedata
    w = 0
    for ch in plain:
        if unicodedata.combining(ch):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def truncate(s: str, width: int, ellipsis: str = "…") -> str:
    """Truncate to a VISIBLE width, keeping any escape sequences intact."""
    if visible_width(s) <= width:
        return s
    out = ""
    seen = 0
    i = 0
    limit = max(0, width - visible_width(ellipsis))
    while i < len(s):
        if s[i] == "\x1b":
            m = ANSI_RE.match(s, i)
            if m:
                out += m.group(0)
                i = m.end()
                continue
        import unicodedata
        ch_w = 2 if unicodedata.east_asian_width(s[i]) in ("W", "F") else 1
        if unicodedata.combining(s[i]):
            ch_w = 0
        if seen + ch_w > limit:
            break
        out += s[i]
        seen += ch_w
        i += 1
    return out + RESET + ellipsis if not _NO_COLOR else out + ellipsis


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - visible_width(s))


# ── frame ────────────────────────────────────────────────────────────────────

def term_width(default: int = 88) -> int:
    try:
        return max(40, min(shutil.get_terminal_size().columns, 140))
    except Exception:
        return default


def term_height(default: int = 30) -> int:
    try:
        return max(12, shutil.get_terminal_size().lines)
    except Exception:
        return default


def clear() -> str:
    return "\x1b[2J\x1b[H"


def home() -> str:
    return "\x1b[H"


def bold_line(title: str, width: int | None = None) -> str:
    w = width or term_width()
    inner = w - 2
    return b("┌" + "─" * inner + "┐") + "\n" + b("│") + " " + pad(b(title), inner - 1) + b("│")


def box(title: str, lines: list[str], width: int | None = None, color: str = "cyan") -> str:
    w = width or term_width()
    inner = w - 2
    out = [style("┌─ " + title + " " + "─" * max(0, inner - visible_width(title) - 4), color, "bold")]
    for ln in lines:
        out.append(style("│", color) + " " + pad(truncate(ln, inner - 2), inner - 1) + style("│", color))
    out.append(style("└" + "─" * inner + "┘", color))
    return "\n".join(out)


def hr(width: int | None = None, ch: str = "─") -> str:
    return dim(ch * (width or term_width()))


# ── keys ─────────────────────────────────────────────────────────────────────

ESC_SEQUENCE_MS = 60
PENDING_KEYS: list[str] = []


def _decode_chunk(chunk: str) -> list[str]:
    """Split one read into individual keys.

    A terminal may deliver several keys in a single read — a paste, a fast double-tap, an
    arrow followed immediately by Enter. Treating the chunk as ONE key silently discards
    the rest, so the CLI ignores input the user definitely made. Longest escape sequence
    first, and iterate by code point so a multi-byte character is one key.
    """
    keys = []
    i = 0
    n = len(chunk)
    seqs = [
        "\x1b[1;5A", "\x1b[1;5B", "\x1b[1;5C", "\x1b[1;5D",
        "\x1b[5~", "\x1b[6~", "\x1b[H", "\x1b[F", "\x1bOA", "\x1bOB",
        "\x1bOC", "\x1bOD", "\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D",
    ]
    while i < n:
        matched = None
        for s in seqs:
            if chunk.startswith(s, i):
                matched = s
                break
        if matched:
            keys.append(matched)
            i += len(matched)
            continue
        keys.append(chunk[i])
        i += 1
    return keys


def read_key(timeout_ms: int | None = None) -> str:
    """Read one keypress, blocking up to timeout_ms (None = wait forever).

    Returns a key name ('up', 'down', 'enter', 'escape', 'backspace', 'tab', 'space',
    'ctrl-c') or the literal character. Pending keys from a multi-key read are queued and
    returned before touching stdin again, so no input is dropped.
    """
    if PENDING_KEYS:
        return PENDING_KEYS.pop(0)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        deadline = None if timeout_ms is None else time.time() + timeout_ms / 1000.0
        chunk = ""
        while True:
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return ""
            else:
                remaining = None
            r, _, _ = select.select([sys.stdin], [], [], remaining)
            if not r:
                return ""
            data = os.read(fd, 64)
            if not data:
                return ""
            chunk += data.decode("utf-8", "replace")
            # A lone ESC is ambiguous: it is either the Escape key or the head of a
            # sequence delivered in a later read. Hold it briefly and see if more arrives.
            if chunk == "\x1b":
                r2, _, _ = select.select([sys.stdin], [], [], ESC_SEQUENCE_MS / 1000.0)
                if r2:
                    chunk += os.read(fd, 64).decode("utf-8", "replace")
                else:
                    return "escape"
            keys = _decode_chunk(chunk)
            if len(keys) == 1 and keys[0] == "\x1b":
                return "escape"
            if keys:
                first = keys.pop(0)
                PENDING_KEYS.extend(keys)
                return _name(first)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _name(k: str) -> str:
    return {
        "\r": "enter", "\n": "enter", "\x7f": "backspace", "\x08": "backspace",
        "\t": "tab", "\x03": "ctrl-c", "\x04": "ctrl-d", "\x1b": "escape",
        "\x1b[A": "up", "\x1b[B": "down", "\x1b[C": "right", "\x1b[D": "left",
        "\x1bOA": "up", "\x1bOB": "down", "\x1bOC": "right", "\x1bOD": "left",
        "\x1b[5~": "pageup", "\x1b[6~": "pagedown",
        "\x1b[H": "home", "\x1b[F": "end",
    }.get(k, k)


def prompt_line(label: str, default: str = "", secret: bool = False) -> str:
    """Read a line of input with echo off during raw mode. Returns the entered text."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    shown = "(set)" if secret and default else default
    buf = ""
    try:
        tty.setraw(fd)
        sys.stdout.write(f"\r{label} {shown}: ")
        sys.stdout.flush()
        while True:
            r, _, _ = select.select([sys.stdin], [], [], None)
            if not r:
                continue
            ch = os.read(fd, 8).decode("utf-8", "replace")
            for c in ch:
                if c in ("\r", "\n"):
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    return buf or default
                if c in ("\x7f", "\x08"):
                    if buf:
                        buf = buf[:-1]
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue
                if c == "\x03":
                    sys.stdout.write("^C\r\n")
                    sys.stdout.flush()
                    raise KeyboardInterrupt
                if c == "\x1b":
                    continue
                buf += c
                sys.stdout.write(c if not secret else "•")
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def confirm(question: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    ans = prompt_line(question + suffix, "")
    if not ans:
        return default
    return ans.strip().lower().startswith("y")


# ── screen ───────────────────────────────────────────────────────────────────

class Screen:
    """Alternate-screen context manager.

    The panel draws and redraws many times; on the primary screen every redraw appends
    and the scrollback fills with stale frames. The alternate buffer makes a redraw
    actually replace the previous one, and the user's scrollback is untouched on exit.
    """

    def __enter__(self):
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[?1049h")   # alternate screen buffer
            sys.stdout.write("\x1b[?25l")     # hide cursor
            sys.stdout.flush()
        return self

    def __exit__(self, *exc):
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[?25h")
            sys.stdout.write("\x1b[?1049l")
            sys.stdout.flush()
        return False


class QuitError(Exception):
    """Raised by a screen when the user asks to leave the whole CLI."""
