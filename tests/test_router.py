from types import SimpleNamespace

import pytest

from jev_router.router import MockTypeSafeClassifier, route_ticket


class FakeClassifier:
    def __init__(self, *, confidence: float, urgent: float, score: float = 2.0):
        self.confidence = confidence
        self.urgent = urgent
        self.score = score
        self.received = None

    def invoke(self, value):
        self.received = value
        return SimpleNamespace(
            choices={
                "department": SimpleNamespace(
                    choice="technical", confidence=self.confidence
                )
            },
            nouls={"urgent": SimpleNamespace(noul=self.urgent)},
            scores={"sentiment": SimpleNamespace(score=self.score)},
        )


def test_auto_routes_a_confident_non_urgent_ticket():
    classifier = FakeClassifier(confidence=0.91, urgent=0.10)

    result = route_ticket("The API returns 500.", classifier=classifier)

    assert result.department == "technical"
    assert result.action == "auto_route"
    assert classifier.received == "The API returns 500."


def test_sends_urgent_ticket_to_priority_queue():
    classifier = FakeClassifier(confidence=0.90, urgent=0.85)

    result = route_ticket("Production is down.", classifier=classifier)

    assert result.action == "priority_queue"


def test_low_confidence_wins_over_urgency_and_requires_review():
    classifier = FakeClassifier(confidence=0.40, urgent=0.95)

    result = route_ticket("Something is wrong.", classifier=classifier)

    assert result.action == "manual_review"


@pytest.mark.parametrize("ticket", ["", "   "])
def test_rejects_empty_ticket(ticket):
    with pytest.raises(ValueError, match="must not be empty"):
        route_ticket(ticket, classifier=FakeClassifier(confidence=1, urgent=0))


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        route_ticket(
            "hello",
            classifier=FakeClassifier(confidence=1, urgent=0),
            min_confidence=threshold,
        )


def test_offline_mock_exercises_the_full_routing_flow():
    result = route_ticket(
        "URGENT: production API is down!",
        classifier=MockTypeSafeClassifier(),
    )

    assert result.department == "technical"
    assert result.department_confidence >= 0.7
    assert result.urgent_probability >= 0.75
    assert result.action == "priority_queue"
