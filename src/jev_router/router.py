"""Classify support tickets with one typed Jev request."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Protocol

from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier


class Classifier(Protocol):
    """The small Runnable surface used by this application."""

    def invoke(self, input: Any) -> Any: ...


class MockTypeSafeClassifier:
    """Offline demo with the same response surface used by the real Runnable.

    This is intentionally a small deterministic heuristic, not a Jev emulator. It lets
    the application and routing policy run before TypeSafe API access is available.
    """

    def invoke(self, input: Any) -> Any:
        text = str(input).lower()
        labels = {
            "billing": ("invoice", "payment", "refund", "subscription", "charge"),
            "technical": ("api", "error", "bug", "down", "connect", "production"),
            "account": ("login", "password", "permission", "account", "access"),
        }
        matches = {
            label: sum(term in text for term in terms)
            for label, terms in labels.items()
        }
        department, match_count = max(matches.items(), key=lambda item: item[1])
        if match_count == 0:
            department, confidence = "general", 0.45
        else:
            confidence = min(0.72 + (match_count - 1) * 0.09, 0.95)

        urgent_terms = ("urgent", "asap", "down", "outage", "security", "blocked")
        urgent = min(sum(term in text for term in urgent_terms) * 0.45, 0.95)
        sentiment_terms = ("frustrated", "angry", "terrible", "unacceptable", "!")
        sentiment = min(sum(term in text for term in sentiment_terms), 3)

        return SimpleNamespace(
            choices={
                "department": SimpleNamespace(
                    choice=department, confidence=confidence
                )
            },
            nouls={"urgent": SimpleNamespace(noul=urgent)},
            scores={"sentiment": SimpleNamespace(score=float(sentiment))},
        )


@dataclass(frozen=True)
class RouteDecision:
    """Stable application output, independent of provider response classes."""

    department: str
    department_confidence: float
    urgent_probability: float
    sentiment_score: float
    action: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def build_classifier() -> TypeSafeClassifier:
    """Build a Runnable that asks all questions in one Jev API call."""

    return TypeSafeClassifier(
        questions={
            "department": Choice(
                instructions="Which support team should own this ticket?",
                criteria={
                    "billing": "Payments, invoices, refunds, plans, or subscriptions.",
                    "technical": "Product failures, bugs, integrations, or API issues.",
                    "account": "Login, profile, permissions, or account access.",
                    "general": "Questions that do not fit the other support teams.",
                },
            ),
            "urgent": Noul(
                instructions=(
                    "Does the ticket require urgent attention due to an outage, "
                    "security risk, blocked critical work, or explicit severe impact?"
                )
            ),
            "sentiment": Score(
                instructions="How frustrated does the customer appear?",
                criteria=["calm", "concerned", "frustrated", "angry"],
            ),
        }
    )


def route_ticket(
    ticket: str,
    *,
    classifier: Classifier | None = None,
    min_confidence: float = 0.70,
    urgent_threshold: float = 0.75,
) -> RouteDecision:
    """Classify a ticket and apply deterministic routing policy.

    Jev supplies probabilistic judgments; regular code owns the decision policy.
    Low-confidence classifications go to review instead of being auto-routed.
    """

    if not ticket.strip():
        raise ValueError("ticket must not be empty")
    if not 0 <= min_confidence <= 1 or not 0 <= urgent_threshold <= 1:
        raise ValueError("thresholds must be between 0 and 1")

    result = (classifier or build_classifier()).invoke(ticket)
    department = result.choices["department"]
    urgent = float(result.nouls["urgent"].noul)
    sentiment = result.scores["sentiment"]
    confidence = float(department.confidence)

    if confidence < min_confidence:
        action = "manual_review"
    elif urgent >= urgent_threshold:
        action = "priority_queue"
    else:
        action = "auto_route"

    return RouteDecision(
        department=str(department.choice),
        department_confidence=round(confidence, 4),
        urgent_probability=round(urgent, 4),
        sentiment_score=round(float(sentiment.score), 4),
        action=action,
    )
