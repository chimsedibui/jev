"""Turn a `GameState` into the text and criteria a Jev `Choice` needs.

This module is the contract between the game and the model. Keeping it separate
means the prompt can be tuned and diffed without touching the rules, and the
same text is what gets printed on screen, so what you debug is what Jev reads.
"""

from __future__ import annotations

from .game import Direction, GameState

HEAD, BODY, FOOD, EMPTY = "@", "o", "*", "."

#: Criteria for a Jev `Choice` question over the four moves. Wording is aimed at
#: a decision model: each option states the consequence, not the geometry.
MOVE_CRITERIA: dict[str, str] = {
    "UP": "Move the head one cell up (toward row 0).",
    "DOWN": "Move the head one cell down (toward the last row).",
    "LEFT": "Move the head one cell left (toward column 0).",
    "RIGHT": "Move the head one cell right (toward the last column).",
}

MOVE_INSTRUCTIONS = (
    "You are steering the snake marked @ on the board below. Choose the next "
    "move. Never pick a move listed as fatal. Prefer the shortest route to the "
    "food marked *, but do not trap the head inside its own body o."
)


def render_board(state: GameState) -> str:
    """ASCII board with a border, head first so overlap never hides the head."""

    grid = [[EMPTY] * state.width for _ in range(state.height)]
    if state.food:
        fx, fy = state.food
        grid[fy][fx] = FOOD
    for x, y in state.snake[1:]:
        grid[y][x] = BODY
    hx, hy = state.head
    grid[hy][hx] = HEAD

    edge = "+" + "-" * state.width + "+"
    rows = ["|" + "".join(row) + "|" for row in grid]
    return "\n".join([edge, *rows, edge])


def _relative_food(state: GameState) -> str:
    if not state.food:
        return "no food left, the board is full"
    hx, hy = state.head
    fx, fy = state.food
    parts = []
    if fy < hy:
        parts.append(f"{hy - fy} up")
    elif fy > hy:
        parts.append(f"{fy - hy} down")
    if fx < hx:
        parts.append(f"{hx - fx} left")
    elif fx > hx:
        parts.append(f"{fx - hx} right")
    return " and ".join(parts) if parts else "on the head cell"


def describe_state(state: GameState) -> str:
    """The exact string handed to Jev.

    Fatal moves are spelled out rather than left for the model to derive from
    coordinates: collision checking is cheap and exact in Python, so the model
    is only asked for the part that needs judgment.
    """

    safe = {d.name for d in state.safe_moves()}
    legal = state.legal_moves()
    lines = []
    for d in Direction:
        if d not in legal:
            lines.append(f"- {d.name}: not allowed, that would reverse into the neck")
        elif d.name in safe:
            lines.append(f"- {d.name}: safe")
        else:
            lines.append(f"- {d.name}: fatal, the head would leave the board or hit the body")

    return "\n".join(
        [
            f"Board {state.width}x{state.height}, column 0 is the left edge and row 0 the top.",
            render_board(state),
            f"Head at {state.head}, moving {state.direction.name}.",
            f"Snake length {len(state.snake)}, score {state.score}, step {state.steps}.",
            f"Food at {state.food}: {_relative_food(state)}.",
            "Move options:",
            *lines,
        ]
    )


def build_move_question():
    """A Jev `Choice` over the four moves.

    Imported lazily so the engine and the heuristic agents stay usable, and
    testable, without `langchain-typesafe` loaded.
    """

    from langchain_typesafe import Choice

    return Choice(instructions=MOVE_INSTRUCTIONS, criteria=dict(MOVE_CRITERIA))
