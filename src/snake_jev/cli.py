"""Command-line entry point: play by hand, or run an agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .agents import GreedyAgent, JevAgent, MockMoveClassifier, RandomAgent
from .game import SnakeGame
from .prompt import build_move_question, describe_state
from .render import KEY_TO_DIRECTION, draw, raw_terminal, read_key
from .runner import benchmark, run_episode

AGENTS = ("greedy", "random", "jev")


def build_agent(args: argparse.Namespace):
    """Turn the CLI flags into an agent, and a factory for repeated episodes."""

    name = getattr(args, "agent", None)
    if name is None:
        return None
    if name == "greedy":
        return GreedyAgent()
    if name == "random":
        return RandomAgent(None)

    if args.mock:
        classifier = MockMoveClassifier()
    else:
        if not os.getenv("TYPESAFE_API_KEY"):
            raise SystemExit(
                "TYPESAFE_API_KEY is missing. Use --mock for an offline run, "
                "or export a key."
            )
        from langchain_typesafe import TypeSafeClassifier

        # Shorter than the 30s default: a stalled move should fall back quickly
        # rather than leave the board frozen.
        classifier = TypeSafeClassifier(
            questions={"move": build_move_question()}, timeout=args.timeout
        )
    return JevAgent(classifier, min_confidence=args.min_confidence)


def _add_agent_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--agent",
        choices=AGENTS,
        default="greedy" if required else None,
        help="Who plays" + ("" if required else "; omit to steer yourself"),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the jev agent offline against a fixed heuristic; no key or credits",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=10.0)


def _add_board_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument(
        "--seed", type=int, default=None, help="Fixed food sequence; replays exactly"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Snake as a decision environment.")
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="Play in the terminal with arrow keys or WASD")
    _add_board_args(play)
    play.add_argument("--tick", type=float, default=0.12, help="Seconds per step")

    auto = sub.add_parser("auto", help="Let an agent play")
    _add_board_args(auto)
    _add_agent_args(auto, required=True)
    auto.add_argument("--episodes", type=int, default=1)
    auto.add_argument("--watch", action="store_true", help="Render each tick")
    auto.add_argument("--fps", type=float, default=20.0)

    ui = sub.add_parser("ui", help="Play in a window (needs the 'ui' extra)")
    _add_board_args(ui)
    ui.add_argument("--tick", type=float, default=0.12, help="Seconds per step")
    ui.add_argument("--cell", type=int, default=28, help="Cell size in pixels")
    _add_agent_args(ui, required=False)

    show = sub.add_parser("show", help="Print the text an agent receives")
    _add_board_args(show)

    return parser


def cmd_play(args: argparse.Namespace) -> int:
    if not sys.stdin.isatty():
        raise SystemExit("play needs a terminal; use 'snake auto' instead")

    game = SnakeGame(args.width, args.height, seed=args.seed)
    footer = "arrows or wasd to steer, q to quit"
    with raw_terminal():
        draw(game.state, footer=footer)
        while not game.state.is_over:
            # Input is polled for exactly one tick, so the snake keeps moving
            # whether or not a key was pressed.
            key = read_key(args.tick)
            if key in ("q", "\x03"):
                break
            game.step(KEY_TO_DIRECTION.get(key or ""))
            draw(game.state, footer=footer)
    print(json.dumps({"score": game.state.score, "steps": game.state.steps}))
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    def factory():
        return build_agent(args)

    if args.watch:
        delay = 1.0 / args.fps if args.fps > 0 else 0.0
        game = SnakeGame(args.width, args.height, seed=args.seed)

        def on_tick(state):
            draw(state, footer=f"agent: {args.agent}")
            time.sleep(delay)

        result = run_episode(game, factory(), on_tick=on_tick)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.episodes == 1:
        game = SnakeGame(args.width, args.height, seed=args.seed)
        agent = factory()
        payload = run_episode(game, agent).to_dict()
        if hasattr(agent, "log"):
            payload["jev"] = {
                "model": agent.model,
                **agent.stats,
                "mean_latency_ms": round(agent.mean_latency_ms, 1),
                "decisions": [d.to_dict() for d in agent.log],
            }
        print(json.dumps(payload, indent=2))
        return 0

    summary = benchmark(
        factory,
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        base_seed=args.seed or 0,
    )
    summary.pop("results")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    # Imported here so every other command still runs without pygame.
    try:
        from .ui import run_ui
    except ImportError as exc:
        raise SystemExit(
            f"the window needs pygame: uv sync --extra ui  ({exc})"
        ) from exc

    game = SnakeGame(args.width, args.height, seed=args.seed)
    state = run_ui(game, agent=build_agent(args), tick=args.tick, cell=args.cell)
    print(json.dumps({"score": state.score, "steps": state.steps}))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    game = SnakeGame(args.width, args.height, seed=args.seed)
    print(describe_state(game.state))
    return 0


def main() -> None:
    args = build_parser().parse_args()
    handlers = {"play": cmd_play, "auto": cmd_auto, "ui": cmd_ui, "show": cmd_show}
    raise SystemExit(handlers[args.command](args))


if __name__ == "__main__":
    main()
