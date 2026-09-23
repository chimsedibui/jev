from dataclasses import replace

import pytest


@pytest.fixture
def place():
    """Force a board layout so one rule can be tested in isolation."""

    def _place(game, snake, food, direction=None):
        game.state = replace(
            game.state,
            snake=tuple(snake),
            food=food,
            direction=direction or game.state.direction,
        )
        return game

    return _place
