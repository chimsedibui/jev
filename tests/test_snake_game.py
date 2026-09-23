import pytest

from snake_jev.game import (
    DEATH_SELF,
    DEATH_STARVED,
    DEATH_WALL,
    STATUS_DEAD,
    STATUS_WON,
    Direction,
    SnakeGame,
)


def test_moving_onto_food_grows_the_snake_and_scores(place):
    game = place(SnakeGame(6, 6, seed=1), [(2, 2), (1, 2)], (3, 2))

    state = game.step(Direction.RIGHT)

    assert state.score == 1
    assert len(state.snake) == 3
    assert state.head == (3, 2)


def test_plain_move_keeps_length_and_drops_the_tail(place):
    game = place(SnakeGame(6, 6, seed=1), [(2, 2), (1, 2)], (5, 5))

    state = game.step(Direction.RIGHT)

    assert state.snake == ((3, 2), (2, 2))
    assert state.score == 0


def test_leaving_the_board_ends_the_episode(place):
    game = place(SnakeGame(6, 6, seed=1), [(0, 2), (1, 2)], (5, 5), Direction.LEFT)

    state = game.step(Direction.LEFT)

    assert state.status == STATUS_DEAD
    assert state.reason == DEATH_WALL


def test_biting_the_body_ends_the_episode(place):
    # (3, 2) is a middle segment, so it is still there when the head arrives.
    snake = [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3)]
    game = place(SnakeGame(6, 6, seed=1), snake, (5, 5), Direction.DOWN)

    state = game.step(Direction.RIGHT)

    assert state.status == STATUS_DEAD
    assert state.reason == DEATH_SELF


def test_the_vacating_tail_cell_is_not_a_collision(place):
    # The tail leaves (3, 2) on the same tick the head arrives there.
    snake = [(2, 2), (2, 1), (3, 1), (3, 2)]
    game = place(SnakeGame(6, 6, seed=1), snake, (5, 5), Direction.DOWN)

    state = game.step(Direction.DOWN)

    assert state.status == "running"
    assert state.head == (2, 3)


def test_reversing_into_the_neck_is_ignored(place):
    game = place(SnakeGame(6, 6, seed=1), [(2, 2), (1, 2)], (5, 5))

    state = game.step(Direction.LEFT)

    assert state.direction is Direction.RIGHT
    assert state.head == (3, 2)


def test_circling_without_eating_eventually_starves():
    game = SnakeGame(5, 5, seed=1, initial_length=1, starvation_limit=6)

    for _ in range(20):
        if game.state.is_over:
            break
        move = next(
            d for d in game.state.safe_moves() if game.state.next_head(d) != game.state.food
        )
        game.step(move)

    assert game.state.status == STATUS_DEAD
    assert game.state.reason == DEATH_STARVED


def test_filling_the_board_wins(place):
    # Eight of nine cells taken, food on the last one.
    snake = [(1, 2), (0, 2), (0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]
    game = place(SnakeGame(3, 3, seed=1), snake, (2, 2), Direction.RIGHT)

    state = game.step(Direction.RIGHT)

    assert state.status == STATUS_WON
    assert state.food is None
    assert len(state.snake) == 9


def test_the_same_seed_replays_the_same_food():
    from snake_jev.agents import GreedyAgent
    from snake_jev.runner import run_episode

    first = run_episode(SnakeGame(8, 8, seed=42), GreedyAgent())
    second = run_episode(SnakeGame(8, 8, seed=42), GreedyAgent())

    assert first.to_dict() == second.to_dict()


def test_different_seeds_differ():
    assert SnakeGame(8, 8, seed=1).state.food != SnakeGame(8, 8, seed=2).state.food


def test_safe_moves_exclude_the_wall(place):
    game = place(SnakeGame(4, 4, seed=1), [(0, 0)], (3, 3))

    assert set(game.state.safe_moves()) == {Direction.DOWN, Direction.RIGHT}


@pytest.mark.parametrize("text,expected", [("up", Direction.UP), ("R", Direction.RIGHT)])
def test_direction_parse_accepts_agent_spelling(text, expected):
    assert Direction.parse(text) is expected


def test_rejects_a_board_too_small_to_play():
    with pytest.raises(ValueError):
        SnakeGame(2, 2)
