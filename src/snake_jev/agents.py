"""Agents that pick a move from a `GameState`.

`GreedyAgent` is the baseline to measure Jev against; `JevAgent` is the seam
where the real model plugs in.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from .game import Direction, GameState, Point
from .prompt import MOVE_CRITERIA, describe_state


class Agent(Protocol):
    """The whole surface the runner needs."""

    name: str

    def next_direction(self, state: GameState) -> Direction: ...


def _fallback(state: GameState) -> Direction:
    """Something legal to return when an agent has no opinion left."""

    safe = state.safe_moves()
    return safe[0] if safe else state.legal_moves()[0]


class RandomAgent:
    """Random among safe moves. The floor any real agent must beat."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def next_direction(self, state: GameState) -> Direction:
        safe = state.safe_moves()
        return self._rng.choice(safe) if safe else _fallback(state)


class GreedyAgent:
    """Shortest path to the food, with a space-preserving fallback.

    When no path exists the snake would otherwise drive into a pocket, so it
    instead takes the move that leaves the most reachable free space.
    """

    name = "greedy"

    def next_direction(self, state: GameState) -> Direction:
        safe = state.safe_moves()
        if not safe:
            return _fallback(state)
        if state.food:
            step = self._first_step_to(state, state.food)
            if step in safe:
                return step
        return max(safe, key=lambda d: self._free_space_after(state, d))

    def _blocked(self, state: GameState) -> set[Point]:
        # The tail moves away this tick, so it is not an obstacle.
        return set(state.snake[:-1])

    def _first_step_to(self, state: GameState, goal: Point) -> Direction | None:
        """BFS from the head; returns the first move of a shortest path."""

        blocked = self._blocked(state)
        start = state.head
        queue: deque[tuple[Point, Direction]] = deque()
        seen = {start}
        for d in state.legal_moves():
            nxt = state.next_head(d)
            if state.in_bounds(nxt) and nxt not in blocked:
                queue.append((nxt, d))
                seen.add(nxt)
        while queue:
            point, first = queue.popleft()
            if point == goal:
                return first
            x, y = point
            for d in Direction:
                dx, dy = d.delta
                nxt = (x + dx, y + dy)
                if nxt in seen or not state.in_bounds(nxt) or nxt in blocked:
                    continue
                seen.add(nxt)
                queue.append((nxt, first))
        return None

    def _free_space_after(self, state: GameState, direction: Direction) -> int:
        """Flood fill the cells still reachable once `direction` is taken."""

        blocked = self._blocked(state)
        start = state.next_head(direction)
        seen = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for d in Direction:
                dx, dy = d.delta
                nxt = (x + dx, y + dy)
                if nxt in seen or not state.in_bounds(nxt) or nxt in blocked:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        return len(seen)


def _short_error(exc: Exception) -> str:
    """A label that fits the log panel and still says what went wrong."""

    detail = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    label = type(exc).__name__
    return f"{label}: {detail[:40]}" if detail and len(detail) <= 40 else label


@dataclass(frozen=True)
class JevDecision:
    """One Jev call, kept so the UI and the logs can show what happened."""

    step: int
    answer: str | None  # what Jev replied, before any policy is applied
    applied: str  # the move actually played
    accepted: bool
    note: str  # why the answer was rejected, empty when accepted
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    safe: tuple[str, ...] = ()  # moves that were survivable when this was asked
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "answer": self.answer,
            "applied": self.applied,
            "accepted": self.accepted,
            "note": self.note,
            "confidence": round(self.confidence, 4) if self.confidence else None,
            "latency_ms": round(self.latency_ms),
        }


class JevAgent:
    """Ask a Jev classifier for the move, then let Python own the policy.

    The classifier is duck-typed on `.invoke(text)` so the game keeps no
    dependency on `langchain-typesafe`; pass a real `TypeSafeClassifier` built
    from `prompt.build_move_question()`, or `MockMoveClassifier` to run offline.

    As in the ticket router, the model supplies a judgment and this code decides
    whether to act on it: a move that is illegal, fatal, or below
    `min_confidence` is replaced by the fallback agent instead of ending the run.
    Every call lands in `log` with the reason, which is what the UI panel reads.
    """

    def __init__(
        self,
        classifier: Any,
        *,
        name: str = "jev",
        question_key: str = "move",
        min_confidence: float = 0.0,
        fallback: Agent | None = None,
        log_size: int = 300,
    ) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.name = name
        self._classifier = classifier
        self._question_key = question_key
        self._min_confidence = min_confidence
        self._fallback = fallback or GreedyAgent()
        #: Newest last. A deque so a long episode cannot grow it without bound.
        self.log: deque[JevDecision] = deque(maxlen=log_size)
        self.model: str | None = None
        self.stats = {
            "calls": 0,
            "fallbacks": 0,
            "errors": 0,
            "latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        }

    @property
    def mean_latency_ms(self) -> float:
        calls = self.stats["calls"]
        return self.stats["latency_ms"] / calls if calls else 0.0

    def next_direction(self, state: GameState) -> Direction:
        self.stats["calls"] += 1
        answer, note, meta = self._ask(state)
        safe = state.safe_moves()

        move = answer
        if move is not None and move not in safe:
            move, note = None, note or "unsafe move"
        if move is None:
            self.stats["fallbacks"] += 1
            move = self._fallback.next_direction(state)

        self.log.append(
            JevDecision(
                step=state.steps,
                answer=answer.name if answer else meta.get("raw"),
                applied=move.name,
                accepted=answer is not None and not note,
                note=note,
                confidence=meta.get("confidence"),
                probabilities=meta.get("probabilities", {}),
                safe=tuple(d.name for d in safe),
                latency_ms=meta.get("latency_ms", 0.0),
                input_tokens=meta.get("input_tokens"),
                output_tokens=meta.get("output_tokens"),
            )
        )
        return move

    def _ask(self, state: GameState) -> tuple[Direction | None, str, dict]:
        """Return (move, rejection note, metadata) for one classifier call.

        Network and validation failures are turned into a note rather than an
        exception: a transient blip should cost one move, not the episode.
        """

        meta: dict = {}
        started = time.perf_counter()
        try:
            result = self._classifier.invoke(describe_state(state))
        except Exception as exc:  # noqa: BLE001 - surfaced in the log, not swallowed
            meta["latency_ms"] = (time.perf_counter() - started) * 1000
            self.stats["errors"] += 1
            self._record(meta)
            return None, _short_error(exc), meta

        meta["latency_ms"] = (time.perf_counter() - started) * 1000
        self.model = getattr(result, "model", None) or self.model
        usage = getattr(result, "usage", None)
        meta["input_tokens"] = getattr(usage, "input_tokens", None)
        meta["output_tokens"] = getattr(usage, "output_tokens", None)
        meta["cost"] = getattr(usage, "cost", None)
        self._record(meta)

        try:
            answer = result.choices[self._question_key]
            meta["raw"] = str(answer.choice)
            meta["probabilities"] = dict(getattr(answer, "probabilities", {}) or {})
            confidence = getattr(answer, "confidence", None)
            meta["confidence"] = float(confidence) if confidence is not None else None
        except (AttributeError, KeyError, TypeError, ValueError):
            return None, "malformed response", meta

        if meta["confidence"] is not None and meta["confidence"] < self._min_confidence:
            return None, "low confidence", meta
        try:
            return Direction.parse(meta["raw"]), "", meta
        except KeyError:
            return None, "unparsable move", meta

    def _record(self, meta: dict) -> None:
        self.stats["latency_ms"] += meta.get("latency_ms", 0.0)
        self.stats["input_tokens"] += meta.get("input_tokens") or 0
        self.stats["output_tokens"] += meta.get("output_tokens") or 0
        self.stats["cost"] += meta.get("cost") or 0.0


class MockMoveClassifier:
    """Offline stand-in with the same response surface as `TypeSafeClassifier`.

    Like the router's mock this is a fixed heuristic, not a Jev emulator: it
    exists so the wiring, the policy and the log panel can be exercised without
    an API key. It reads the prompt text, never the `GameState`, so a prompt
    that omits something the model would need shows up here too.
    """

    model = "mock-jev"

    def __init__(self, seed: int | None = 0, noise: float = 0.6) -> None:
        self._rng = random.Random(seed)
        self._noise = noise

    def invoke(self, text: str) -> Any:
        from types import SimpleNamespace

        fatal = {
            name
            for name in MOVE_CRITERIA
            if f"- {name}: fatal" in text or f"- {name}: not allowed" in text
        }
        toward = self._toward(text)

        logits = {}
        for name in MOVE_CRITERIA:
            score = -6.0 if name in fatal else 0.0
            if name in toward:
                score += 2.0
            logits[name] = score + self._rng.gauss(0, self._noise)

        top = max(logits.values())
        exp = {k: math.exp(v - top) for k, v in logits.items()}
        total = sum(exp.values())
        probabilities = {k: v / total for k, v in exp.items()}
        choice = max(probabilities, key=probabilities.get)

        return SimpleNamespace(
            model=self.model,
            usage=SimpleNamespace(input_tokens=len(text) // 4, output_tokens=4),
            request_id=None,
            choices={
                "move": SimpleNamespace(
                    choice=choice,
                    probabilities=probabilities,
                    confidence=concentration(probabilities),
                )
            },
        )

    @staticmethod
    def _toward(text: str) -> set[str]:
        """Directions the prompt says the food lies in."""

        line = next((l for l in text.splitlines() if l.startswith("Food at")), "")
        return {name for name in MOVE_CRITERIA if name.lower() in line.lower()}


def concentration(probabilities: dict[str, float]) -> float:
    """How peaked a distribution is, on a 0..1 scale, like TypeSafe's confidence."""

    n = len(probabilities)
    if n < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
    return max(0.0, min(1.0, 1 - entropy / math.log(n)))
