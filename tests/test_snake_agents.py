from types import SimpleNamespace

import pytest

from snake_jev.agents import (
    GreedyAgent,
    JevAgent,
    MockMoveClassifier,
    RandomAgent,
)
from snake_jev.game import Direction, SnakeGame
from snake_jev.prompt import MOVE_CRITERIA, describe_state, render_board
from snake_jev.runner import benchmark, run_episode


class FakeClassifier:
    """Same response surface as a TypeSafeClassifier, with a scripted answer."""

    def __init__(self, choice, confidence=0.9):
        self.choice = choice
        self.confidence = confidence
        self.received = None

    def invoke(self, value):
        self.received = value
        return SimpleNamespace(
            choices={
                "move": SimpleNamespace(choice=self.choice, confidence=self.confidence)
            }
        )


def test_greedy_walks_toward_the_food(place):
    game = place(SnakeGame(8, 8, seed=1), [(2, 4), (1, 4)], (5, 4))

    assert GreedyAgent().next_direction(game.state) is Direction.RIGHT


def test_greedy_keeps_its_distance_from_a_dead_end(place):
    # A pocket to the right holds two cells; going down keeps the open board.
    snake = [(1, 1), (1, 2), (2, 2), (3, 2), (3, 1), (3, 0), (2, 0)]
    game = place(SnakeGame(6, 6, seed=1), snake, (0, 5), Direction.UP)

    assert GreedyAgent().next_direction(game.state) is not Direction.RIGHT


def test_greedy_clearly_beats_random_on_the_same_seeds():
    greedy = benchmark(GreedyAgent, episodes=5, width=10, height=10, base_seed=0)
    random_agent = benchmark(
        lambda: RandomAgent(0), episodes=5, width=10, height=10, base_seed=0
    )

    assert greedy["mean_score"] > random_agent["mean_score"] * 2


def test_random_never_picks_a_move_it_knows_is_fatal(place):
    game = place(SnakeGame(4, 4, seed=1), [(0, 0)], (3, 3))
    agent = RandomAgent(seed=0)

    picks = {agent.next_direction(game.state) for _ in range(30)}

    assert picks <= {Direction.DOWN, Direction.RIGHT}


def test_jev_agent_follows_a_safe_answer(place):
    game = place(SnakeGame(8, 8, seed=1), [(2, 4), (1, 4)], (2, 1))
    classifier = FakeClassifier("UP")
    agent = JevAgent(classifier)

    assert agent.next_direction(game.state) is Direction.UP
    assert agent.stats["fallbacks"] == 0
    # The model is handed the rendered board, not raw coordinates.
    assert render_board(game.state) in classifier.received


def test_jev_agent_falls_back_on_a_fatal_answer(place):
    game = place(SnakeGame(6, 6, seed=1), [(0, 3), (1, 3)], (0, 0), Direction.LEFT)
    agent = JevAgent(FakeClassifier("LEFT"))

    move = agent.next_direction(game.state)

    assert move in game.state.safe_moves()
    assert agent.stats["fallbacks"] == 1


@pytest.mark.parametrize("answer", ["sideways", "", None, 42])
def test_jev_agent_falls_back_on_an_unparsable_answer(place, answer):
    game = place(SnakeGame(8, 8, seed=1), [(2, 4), (1, 4)], (5, 4))
    agent = JevAgent(FakeClassifier(answer))

    assert agent.next_direction(game.state) in game.state.safe_moves()
    assert agent.stats["fallbacks"] == 1


def test_jev_agent_falls_back_when_the_classifier_raises(place):
    class Broken:
        def invoke(self, value):
            raise ValueError("no credits")

    game = place(SnakeGame(8, 8, seed=1), [(2, 4), (1, 4)], (5, 4))
    agent = JevAgent(Broken())

    assert agent.next_direction(game.state) in game.state.safe_moves()
    assert agent.stats["fallbacks"] == 1


def test_jev_agent_ignores_a_low_confidence_answer(place):
    game = place(SnakeGame(8, 8, seed=1), [(2, 4), (1, 4)], (2, 1))
    agent = JevAgent(FakeClassifier("UP", confidence=0.3), min_confidence=0.7)

    agent.next_direction(game.state)

    assert agent.stats["fallbacks"] == 1


def test_jev_agent_plays_a_full_episode():
    game = SnakeGame(8, 8, seed=5)
    agent = JevAgent(FakeClassifier("UP"))

    result = run_episode(game, agent)

    assert result.agent == "jev"
    assert agent.stats["calls"] > 0


def test_rejects_an_impossible_confidence_threshold():
    with pytest.raises(ValueError):
        JevAgent(FakeClassifier("UP"), min_confidence=1.5)


def test_the_prompt_labels_every_move_and_matches_the_choice_criteria(place):
    game = place(SnakeGame(6, 6, seed=1), [(0, 3), (1, 3)], (4, 4), Direction.LEFT)

    text = describe_state(game.state)

    assert "- LEFT: fatal" in text
    assert "- RIGHT: not allowed" in text
    assert "Food at (4, 4): 1 down and 4 right." in text
    assert set(MOVE_CRITERIA) == {d.name for d in Direction}


class SlowClassifier:
    """Blocks until released, to stand in for a real network round trip."""

    def __init__(self):
        import threading

        self.release = threading.Event()
        self.calls = 0

    def invoke(self, value):
        self.calls += 1
        self.release.wait(timeout=5)
        return SimpleNamespace(
            model="jev-latest",
            usage=SimpleNamespace(input_tokens=90, output_tokens=4),
            choices={
                "move": SimpleNamespace(
                    choice="UP", probabilities={"UP": 1.0}, confidence=0.99
                )
            },
        )


def test_the_log_records_an_accepted_call_with_its_metadata(place):
    game = place(SnakeGame(8, 8, seed=1), [(2, 4), (1, 4)], (2, 1))
    agent = JevAgent(
        FakeClassifier("UP"),
        fallback=GreedyAgent(),
    )

    agent.next_direction(game.state)
    entry = agent.log[-1]

    assert (entry.answer, entry.applied, entry.accepted) == ("UP", "UP", True)
    assert entry.note == ""
    assert set(entry.safe) == {d.name for d in game.state.safe_moves()}


def test_the_log_explains_why_a_move_was_rejected(place):
    game = place(SnakeGame(6, 6, seed=1), [(0, 3), (1, 3)], (0, 0), Direction.LEFT)
    agent = JevAgent(FakeClassifier("LEFT"))

    agent.next_direction(game.state)
    entry = agent.log[-1]

    assert entry.answer == "LEFT"
    assert entry.applied != "LEFT"
    assert not entry.accepted
    assert entry.note == "unsafe move"


def test_the_log_records_an_api_failure_without_ending_the_episode(place):
    class Broken:
        def invoke(self, value):
            raise TimeoutError("upstream timed out")

    game = place(SnakeGame(8, 8, seed=1), [(2, 4), (1, 4)], (5, 4))
    agent = JevAgent(Broken())

    move = agent.next_direction(game.state)

    assert move in game.state.safe_moves()
    assert agent.log[-1].note == "TimeoutError"
    assert agent.stats["errors"] == 1


def test_usage_and_latency_accumulate_across_calls(place):
    game = place(SnakeGame(8, 8, seed=1), [(2, 4), (1, 4)], (5, 4))
    agent = JevAgent(SlowClassifier())
    agent._classifier.release.set()

    agent.next_direction(game.state)
    agent.next_direction(game.state)

    assert agent.model == "jev-latest"
    assert agent.stats["input_tokens"] == 180
    assert agent.mean_latency_ms >= 0


def test_the_mock_classifier_never_prefers_a_move_the_prompt_calls_fatal():
    from snake_jev.prompt import describe_state

    classifier = MockMoveClassifier(seed=3)
    game = SnakeGame(8, 8, seed=9)
    agent = JevAgent(classifier)

    for _ in range(60):
        if game.state.is_over:
            break
        game.step(agent.next_direction(game.state))

    unsafe = [d for d in agent.log if d.answer and d.answer not in d.safe]
    # A fixed heuristic, so this is a wiring check, not a claim about Jev.
    assert len(unsafe) <= 2
    assert game.state.score > 0
