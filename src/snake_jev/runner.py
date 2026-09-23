"""Drive an agent through episodes and report comparable numbers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from .agents import Agent
from .game import GameState, SnakeGame


@dataclass(frozen=True)
class EpisodeResult:
    """One finished episode, in the terms used to compare agents."""

    agent: str
    seed: int | None
    score: int
    steps: int
    status: str
    reason: str | None
    board: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(
    game: SnakeGame,
    agent: Agent,
    *,
    max_steps: int | None = None,
    on_tick: Callable[[GameState], None] | None = None,
) -> EpisodeResult:
    """Play one episode to its end.

    `max_steps` is a second guard on top of the engine's starvation limit, for
    an agent that is slow per move rather than merely indecisive.
    """

    state = game.state
    if on_tick:
        on_tick(state)
    while not state.is_over:
        if max_steps is not None and state.steps >= max_steps:
            break
        state = game.step(agent.next_direction(state))
        if on_tick:
            on_tick(state)

    return EpisodeResult(
        agent=agent.name,
        seed=game.seed,
        score=state.score,
        steps=state.steps,
        status=state.status,
        reason=state.reason,
        board=f"{state.width}x{state.height}",
    )


def benchmark(
    agent_factory: Callable[[], Agent],
    *,
    episodes: int = 10,
    width: int = 12,
    height: int = 12,
    base_seed: int = 0,
    max_steps: int | None = None,
) -> dict:
    """Run consecutive seeds so two agents can be compared on identical food.

    A fresh agent per episode keeps any internal state from leaking across runs.
    """

    results = [
        run_episode(
            SnakeGame(width, height, seed=base_seed + i),
            agent_factory(),
            max_steps=max_steps,
        )
        for i in range(episodes)
    ]
    scores = [r.score for r in results]
    deaths: dict[str, int] = {}
    for r in results:
        key = r.reason or r.status
        deaths[key] = deaths.get(key, 0) + 1

    return {
        "agent": results[0].agent,
        "board": f"{width}x{height}",
        "episodes": episodes,
        "mean_score": round(sum(scores) / len(scores), 2),
        "best_score": max(scores),
        "worst_score": min(scores),
        "mean_steps": round(sum(r.steps for r in results) / len(results), 1),
        "outcomes": deaths,
        "results": [r.to_dict() for r in results],
    }
