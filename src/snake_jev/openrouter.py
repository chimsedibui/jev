"""Drive the snake with a chat model through OpenRouter.

This is not Jev. Jev is a decision model that returns a calibrated distribution
over the labels you give it; a chat model returns text. To keep the same
response surface honestly, the distribution here is read from the model's own
`top_logprobs` over the first generated token, never from asking the model to
score its own certainty.

Only the standard library is used, so the game keeps no HTTP dependency.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from types import SimpleNamespace

from .agents import concentration
from .prompt import MOVE_CRITERIA, MOVE_INSTRUCTIONS

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = (
    MOVE_INSTRUCTIONS
    + " Reply with exactly one word and nothing else: UP, DOWN, LEFT or RIGHT."
)


class OpenRouterError(RuntimeError):
    """Base class, so `JevAgent` can log a short, specific reason."""


class OpenRouterAuthError(OpenRouterError):
    """The key was rejected."""


class OpenRouterRateLimited(OpenRouterError):
    """Too many requests, or credits exhausted."""


class OpenRouterBadResponse(OpenRouterError):
    """A 200 that did not contain a usable answer."""


class OpenRouterMoveClassifier:
    """`TypeSafeClassifier`-shaped wrapper around an OpenRouter chat model.

    Returns the same surface `JevAgent` expects: `.model`, `.usage`, and
    `.choices["move"]` with `choice`, `probabilities` and `confidence`.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 20.0,
        top_logprobs: int = 10,
    ) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise OpenRouterAuthError("OPENROUTER_API_KEY is not set")
        self._key = key
        self.model = model
        self.timeout = timeout
        self.top_logprobs = top_logprobs

    def invoke(self, text: str) -> SimpleNamespace:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            # One word is all that is wanted; a longer reply is a malformed one.
            "max_tokens": 3,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        data = self._post(payload)
        return self._parse(data)

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "X-Title": "snake-jev",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise OpenRouterAuthError(f"HTTP {exc.code}") from exc
            if exc.code == 429:
                raise OpenRouterRateLimited("HTTP 429") from exc
            raise OpenRouterError(f"HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise OpenRouterError(f"connection failed: {exc.reason}") from exc

    def _parse(self, data: dict) -> SimpleNamespace:
        try:
            choice = data["choices"][0]
            content = (choice["message"]["content"] or "").strip().upper()
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterBadResponse("no message in response") from exc

        probabilities = _distribution(choice.get("logprobs"))
        if content in MOVE_CRITERIA:
            answer = content
        elif probabilities:
            # The reply was not one of the four words, but the token
            # distribution still says which one it was leaning towards.
            answer = max(probabilities, key=probabilities.get)
        else:
            raise OpenRouterBadResponse(f"unusable reply {content[:12]!r}")

        if not probabilities:
            probabilities = {name: float(name == answer) for name in MOVE_CRITERIA}

        usage = data.get("usage") or {}
        return SimpleNamespace(
            model=data.get("model", self.model),
            request_id=data.get("id"),
            usage=SimpleNamespace(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                cost=usage.get("cost"),
            ),
            choices={
                "move": SimpleNamespace(
                    choice=answer,
                    probabilities=probabilities,
                    confidence=concentration(probabilities),
                )
            },
        )


def _distribution(logprobs: dict | None) -> dict[str, float]:
    """Fold the first token's `top_logprobs` onto the four move labels.

    Several tokens can spell the same move (`UP`, ` UP`, `Up`), so their
    probabilities are summed before the result is renormalised over the labels
    that matter. An empty dict means the model did not return logprobs.
    """

    entries = (logprobs or {}).get("content") or []
    if not entries:
        return {}
    totals = {name: 0.0 for name in MOVE_CRITERIA}
    for candidate in entries[0].get("top_logprobs") or []:
        name = str(candidate.get("token", "")).strip().upper()
        if name in totals:
            totals[name] += math.exp(candidate["logprob"])
    total = sum(totals.values())
    if total <= 0:
        return {}
    return {name: value / total for name, value in totals.items()}
