"""Terminal rendering and keyboard input for the human-played mode."""

from __future__ import annotations

import select
import sys
import termios
import tty
from contextlib import contextmanager

from .game import Direction, GameState
from .prompt import render_board

CLEAR = "\x1b[H\x1b[2J"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"

KEY_TO_DIRECTION = {
    "\x1b[A": Direction.UP,
    "\x1b[B": Direction.DOWN,
    "\x1b[D": Direction.LEFT,
    "\x1b[C": Direction.RIGHT,
    "w": Direction.UP,
    "s": Direction.DOWN,
    "a": Direction.LEFT,
    "d": Direction.RIGHT,
}


def frame(state: GameState, *, footer: str = "") -> str:
    """One screen: board plus the same numbers the agents see."""

    header = (
        f"score {state.score}   length {len(state.snake)}   step {state.steps}"
    )
    lines = [header, render_board(state)]
    if state.is_over:
        lines.append(f"game over: {state.reason or state.status}")
    if footer:
        lines.append(footer)
    return "\n".join(lines)


def draw(state: GameState, *, footer: str = "") -> None:
    sys.stdout.write(CLEAR + frame(state, footer=footer) + "\n")
    sys.stdout.flush()


@contextmanager
def raw_terminal():
    """Read keys unbuffered, and restore the terminal whatever happens."""

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


def read_key(timeout: float) -> str | None:
    """Return the key pressed within `timeout` seconds, or None.

    Arrow keys arrive as a three-byte escape sequence, so the rest of it is
    drained immediately rather than treated as separate keystrokes.
    """

    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    key = sys.stdin.read(1)
    if key == "\x1b":
        while select.select([sys.stdin], [], [], 0)[0]:
            key += sys.stdin.read(1)
            if len(key) >= 3:
                break
    return key
