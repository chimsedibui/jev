import os
import subprocess
import sys
import time

import pytest

from snake_jev.game import Direction, SnakeGame


def test_importing_the_package_does_not_pull_in_pygame():
    """The engine, the agents and the tests must run without the 'ui' extra."""

    code = "import snake_jev, sys; print('pygame' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )
    assert out.stdout.strip() == "False"


@pytest.fixture
def ui_factory():
    pygame = pytest.importorskip("pygame")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    created = []

    def _make(**kwargs):
        from snake_jev.ui import SnakeUI

        ui = SnakeUI(SnakeGame(10, 10, seed=1), **kwargs)
        created.append(ui)
        return ui

    yield _make
    for _ in created:
        pygame.quit()


def test_a_frame_renders_something(ui_factory):
    ui = ui_factory()

    ui._draw()

    # A blank fill would leave every pixel on the background colour.
    colours = {ui.screen.get_at((x, y))[:3] for x in range(0, 300, 7) for y in range(0, 300, 7)}
    assert len(colours) > 3


def test_queued_keys_are_applied_one_per_tick(ui_factory):
    ui = ui_factory()
    ui.pending.extend([Direction.UP, Direction.LEFT])

    ui._advance()
    assert ui.game.state.direction is Direction.UP

    ui._advance()
    assert ui.game.state.direction is Direction.LEFT


def test_a_reversing_key_is_dropped_rather_than_stalling_the_queue(ui_factory):
    ui = ui_factory()  # starts moving RIGHT
    ui.pending.extend([Direction.LEFT, Direction.UP])

    ui._advance()

    assert ui.game.state.direction is Direction.UP


def test_the_agent_steers_when_one_is_given(ui_factory):
    from snake_jev.agents import GreedyAgent

    ui = ui_factory(agent=GreedyAgent())
    ui.pending.append(Direction.UP)  # ignored: the agent is in charge

    for _ in range(12):
        ui._advance()

    assert ui.game.state.steps == 12
    assert ui.game.state.score > 0 or not ui.game.state.is_over


def test_restart_clears_the_board_and_the_queue(ui_factory):
    ui = ui_factory()
    for _ in range(5):
        ui._advance()
    ui.pending.append(Direction.UP)

    ui._restart()

    assert ui.game.state.steps == 0
    assert not ui.pending
    assert ui.prev_state is None


def test_a_slow_agent_does_not_block_the_render_loop(ui_factory):
    from tests.test_snake_agents import SlowClassifier
    from snake_jev.agents import JevAgent

    classifier = SlowClassifier()
    ui = ui_factory(agent=JevAgent(classifier), tick=0.0)

    ui.accumulator = 1.0
    ui._pump_agent()                 # submits, returns immediately
    assert ui._thinking is not None
    ui._draw()                       # the window still paints while waiting
    assert ui.game.state.steps == 0

    classifier.release.set()
    for _ in range(200):
        ui._pump_agent()
        if ui.game.state.steps:
            break
        time.sleep(0.01)

    assert ui.game.state.steps == 1
    assert ui._thinking is None


def test_the_panel_only_appears_for_an_agent_that_logs(ui_factory):
    from snake_jev.agents import GreedyAgent, JevAgent, MockMoveClassifier

    plain = ui_factory(agent=GreedyAgent())
    assert not plain.log_panel
    narrow = plain.screen.get_width()

    logged = ui_factory(agent=JevAgent(MockMoveClassifier()))
    assert logged.log_panel
    assert logged.screen.get_width() > narrow


def test_the_panel_renders_a_logged_decision(ui_factory):
    from snake_jev.agents import JevAgent, MockMoveClassifier

    ui = ui_factory(agent=JevAgent(MockMoveClassifier(seed=1)))
    for _ in range(5):
        ui._advance()

    ui._draw()

    panel = ui._panel_rect()
    colours = {
        ui.screen.get_at((x, y))[:3]
        for x in range(panel.x + 5, panel.right - 5, 6)
        for y in range(panel.y + 5, panel.bottom - 5, 6)
    }
    assert len(colours) > 5
    assert ui.agent.log
