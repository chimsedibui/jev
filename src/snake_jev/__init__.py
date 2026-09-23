"""Snake as a decision environment for Jev."""

from .agents import (
    Agent,
    GreedyAgent,
    JevAgent,
    JevDecision,
    MockMoveClassifier,
    RandomAgent,
)
from .game import Direction, GameState, SnakeGame
from .prompt import MOVE_CRITERIA, build_move_question, describe_state, render_board
from .runner import EpisodeResult, benchmark, run_episode

__all__ = [
    "Agent",
    "Direction",
    "EpisodeResult",
    "GameState",
    "GreedyAgent",
    "JevDecision",
    "JevAgent",
    "MOVE_CRITERIA",
    "MockMoveClassifier",
    "RandomAgent",
    "SnakeGame",
    "benchmark",
    "build_move_question",
    "describe_state",
    "render_board",
    "run_episode",
]
