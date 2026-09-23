"""Pure snake engine: no I/O, no rendering, no randomness that cannot be seeded.

The engine exists so an agent (human, heuristic, or Jev) can be dropped in
without touching game rules. Every episode is reproducible from `seed`, which is
what makes it possible to compare two agents on the same food sequence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum

Point = tuple[int, int]


class Direction(Enum):
    """A move, stored as the (dx, dy) delta applied to the head."""

    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def delta(self) -> Point:
        return self.value

    @property
    def opposite(self) -> "Direction":
        dx, dy = self.value
        return Direction((-dx, -dy))

    @classmethod
    def parse(cls, name: str) -> "Direction":
        """Accept the names an agent is likely to emit ('up', 'UP', 'u')."""

        key = name.strip().upper()
        aliases = {"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT"}
        return cls[aliases.get(key, key)]


DEATH_WALL = "hit_wall"
DEATH_SELF = "hit_self"
DEATH_STARVED = "starved"

STATUS_RUNNING = "running"
STATUS_DEAD = "dead"
STATUS_WON = "won"


@dataclass(frozen=True)
class GameState:
    """An immutable snapshot. Agents only ever see this."""

    width: int
    height: int
    snake: tuple[Point, ...]  # head first, tail last
    food: Point | None
    direction: Direction
    score: int
    steps: int
    steps_since_food: int
    starvation_limit: int
    status: str = STATUS_RUNNING
    reason: str | None = None

    @property
    def head(self) -> Point:
        return self.snake[0]

    @property
    def tail(self) -> Point:
        return self.snake[-1]

    @property
    def is_over(self) -> bool:
        return self.status != STATUS_RUNNING

    def in_bounds(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def next_head(self, direction: Direction) -> Point:
        dx, dy = direction.delta
        x, y = self.head
        return (x + dx, y + dy)

    def legal_moves(self) -> tuple[Direction, ...]:
        """Moves the engine will not silently rewrite (no 180-degree turn)."""

        if len(self.snake) == 1:
            return tuple(Direction)
        return tuple(d for d in Direction if d is not self.direction.opposite)

    def would_collide(self, direction: Direction) -> bool:
        """True if taking `direction` now ends the episode.

        The tail cell is free unless the snake is about to grow into it, because
        the tail vacates on the same tick.
        """

        target = self.next_head(direction)
        if not self.in_bounds(target):
            return True
        grows = target == self.food
        body = self.snake if grows else self.snake[:-1]
        return target in body

    def safe_moves(self) -> tuple[Direction, ...]:
        """Legal moves that survive at least one tick."""

        return tuple(d for d in self.legal_moves() if not self.would_collide(d))

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "snake": [list(p) for p in self.snake],
            "food": list(self.food) if self.food else None,
            "direction": self.direction.name,
            "score": self.score,
            "steps": self.steps,
            "status": self.status,
            "reason": self.reason,
        }


class SnakeGame:
    """Mutable driver around `GameState`."""

    def __init__(
        self,
        width: int = 12,
        height: int = 12,
        *,
        seed: int | None = None,
        initial_length: int = 3,
        starvation_limit: int | None = None,
    ) -> None:
        if width < 3 or height < 3:
            raise ValueError("board must be at least 3x3")
        if not 1 <= initial_length <= width:
            raise ValueError("initial_length must fit on one row")

        self.width = width
        self.height = height
        self.seed = seed
        self.initial_length = initial_length
        # Without this an agent that circles forever would never end an episode.
        self.starvation_limit = starvation_limit or width * height * 2
        self.reset()

    def reset(self) -> GameState:
        self._rng = random.Random(self.seed)
        y = self.height // 2
        x = self.width // 2
        snake = tuple((x - i, y) for i in range(self.initial_length))
        self.state = GameState(
            width=self.width,
            height=self.height,
            snake=snake,
            food=None,
            direction=Direction.RIGHT,
            score=0,
            steps=0,
            steps_since_food=0,
            starvation_limit=self.starvation_limit,
        )
        self.state = replace(self.state, food=self._spawn_food(snake))
        return self.state

    def _spawn_food(self, snake: tuple[Point, ...]) -> Point | None:
        occupied = set(snake)
        free = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not free:
            return None
        return self._rng.choice(free)

    def step(self, direction: Direction | None = None) -> GameState:
        """Advance one tick. A 180-degree turn is ignored, as in the arcade game."""

        state = self.state
        if state.is_over:
            return state

        move = direction or state.direction
        if len(state.snake) > 1 and move is state.direction.opposite:
            move = state.direction

        target = state.next_head(move)
        grows = target == state.food
        body = state.snake if grows else state.snake[:-1]

        if not state.in_bounds(target):
            return self._die(move, DEATH_WALL)
        if target in body:
            return self._die(move, DEATH_SELF)

        snake = (target,) + body
        score = state.score + (1 if grows else 0)
        food = self._spawn_food(snake) if grows else state.food
        steps_since_food = 0 if grows else state.steps_since_food + 1

        self.state = replace(
            state,
            snake=snake,
            food=food,
            direction=move,
            score=score,
            steps=state.steps + 1,
            steps_since_food=steps_since_food,
        )
        if food is None:
            self.state = replace(self.state, status=STATUS_WON)
        elif steps_since_food > state.starvation_limit:
            self.state = replace(
                self.state, status=STATUS_DEAD, reason=DEATH_STARVED
            )
        return self.state

    def _die(self, move: Direction, reason: str) -> GameState:
        self.state = replace(
            self.state,
            direction=move,
            steps=self.state.steps + 1,
            status=STATUS_DEAD,
            reason=reason,
        )
        return self.state
