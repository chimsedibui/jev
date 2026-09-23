import io
import json
import urllib.error

import pytest

from snake_jev.openrouter import (
    OpenRouterAuthError,
    OpenRouterBadResponse,
    OpenRouterError,
    OpenRouterMoveClassifier,
    OpenRouterRateLimited,
    _distribution,
)


def logprobs(*pairs):
    return {"content": [{"top_logprobs": [{"token": t, "logprob": lp} for t, lp in pairs]}]}


def response(content, lp=None, usage=None):
    return {
        "model": "openai/gpt-4o-mini",
        "id": "gen-1",
        "usage": usage or {"prompt_tokens": 196, "completion_tokens": 1, "cost": 3e-05},
        "choices": [{"message": {"content": content}, "logprobs": lp}],
    }


@pytest.fixture
def classifier():
    return OpenRouterMoveClassifier(api_key="sk-or-v1-test")


def test_spellings_of_the_same_move_are_summed():
    dist = _distribution(logprobs(("UP", -1.0), (" UP", -1.0), ("DOWN", -1.0), ("Up", -1.0)))

    assert dist["UP"] == pytest.approx(0.75)
    assert dist["DOWN"] == pytest.approx(0.25)


def test_tokens_that_are_not_moves_are_dropped_and_the_rest_renormalised():
    dist = _distribution(logprobs(("UP", -0.1), ("the", -0.1), ("DOWN", -2.0)))

    assert set(dist) == {"UP", "DOWN", "LEFT", "RIGHT"}
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["UP"] > dist["DOWN"]
    assert dist["LEFT"] == 0.0


def test_a_model_without_logprobs_yields_no_distribution():
    assert _distribution(None) == {}
    assert _distribution({"content": []}) == {}


def test_the_reply_is_used_when_it_names_a_move(classifier):
    answer = classifier._parse(response("UP", logprobs(("UP", -0.1), ("DOWN", -2.0))))

    move = answer.choices["move"]
    assert move.choice == "UP"
    assert move.probabilities["UP"] > move.probabilities["DOWN"]
    assert 0 <= move.confidence <= 1
    assert answer.usage.input_tokens == 196
    assert answer.usage.cost == 3e-05


def test_a_chatty_reply_falls_back_to_the_token_distribution(classifier):
    answer = classifier._parse(response("Well, I think", logprobs(("DOWN", -0.1), ("UP", -3.0))))

    assert answer.choices["move"].choice == "DOWN"


def test_a_reply_with_neither_a_move_nor_logprobs_is_rejected(classifier):
    with pytest.raises(OpenRouterBadResponse):
        classifier._parse(response("no idea"))


def test_a_bare_reply_without_logprobs_still_works(classifier):
    answer = classifier._parse(response("LEFT"))

    move = answer.choices["move"]
    assert move.choice == "LEFT"
    assert move.probabilities == {"UP": 0.0, "DOWN": 0.0, "LEFT": 1.0, "RIGHT": 0.0}


def test_a_missing_key_is_reported_before_any_request(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(OpenRouterAuthError):
        OpenRouterMoveClassifier()


@pytest.mark.parametrize(
    "code,expected",
    [(401, OpenRouterAuthError), (429, OpenRouterRateLimited), (500, OpenRouterError)],
)
def test_http_failures_map_to_specific_errors(classifier, monkeypatch, code, expected):
    def raise_http(*args, **kwargs):
        raise urllib.error.HTTPError("url", code, "boom", {}, io.BytesIO(b""))

    monkeypatch.setattr("snake_jev.openrouter.urllib.request.urlopen", raise_http)

    with pytest.raises(expected):
        classifier.invoke("board")


def test_a_dropped_connection_is_reported_as_one_error(classifier, monkeypatch):
    def raise_url(*args, **kwargs):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr("snake_jev.openrouter.urllib.request.urlopen", raise_url)

    with pytest.raises(OpenRouterError):
        classifier.invoke("board")


def test_the_request_asks_for_one_word_and_its_logprobs(classifier, monkeypatch):
    sent = {}

    class FakeResponse:
        def __enter__(self):
            return io.BytesIO(json.dumps(response("UP", logprobs(("UP", -0.1)))).encode())

        def __exit__(self, *exc):
            return False

    def capture(request, timeout=None):
        sent["body"] = json.loads(request.data)
        sent["auth"] = request.headers["Authorization"]
        return FakeResponse()

    monkeypatch.setattr("snake_jev.openrouter.urllib.request.urlopen", capture)
    classifier.invoke("the board")

    assert sent["body"]["logprobs"] is True
    assert sent["body"]["max_tokens"] == 3
    assert sent["body"]["temperature"] == 0
    assert sent["auth"].startswith("Bearer ")
    assert sent["body"]["messages"][1]["content"] == "the board"


def test_the_agent_logs_an_auth_failure_and_keeps_playing(monkeypatch):
    from snake_jev.agents import JevAgent
    from snake_jev.game import SnakeGame

    def raise_http(*args, **kwargs):
        raise urllib.error.HTTPError("url", 401, "no", {}, io.BytesIO(b""))

    monkeypatch.setattr("snake_jev.openrouter.urllib.request.urlopen", raise_http)
    game = SnakeGame(8, 8, seed=1)
    agent = JevAgent(
        OpenRouterMoveClassifier(api_key="sk-or-v1-test"), name="openrouter"
    )

    move = agent.next_direction(game.state)

    assert move in game.state.safe_moves()
    assert agent.log[-1].note == "OpenRouterAuthError: HTTP 401"
    assert agent.name == "openrouter"
