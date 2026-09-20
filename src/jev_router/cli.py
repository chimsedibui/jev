"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import os

from .router import MockTypeSafeClassifier, route_ticket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route a support ticket with TypeSafe Jev."
    )
    parser.add_argument("ticket", help="Customer support ticket text")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--urgent-threshold", type=float, default=0.75)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline with deterministic demo data; no API key or credits needed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.mock and not os.getenv("TYPESAFE_API_KEY"):
        raise SystemExit(
            "TYPESAFE_API_KEY is missing. Use --mock for an offline demo, or export a key."
        )

    decision = route_ticket(
        args.ticket,
        classifier=MockTypeSafeClassifier() if args.mock else None,
        min_confidence=args.min_confidence,
        urgent_threshold=args.urgent_threshold,
    )
    print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
