"""Pygame window for the same engine the terminal and the agents drive.

Rendering only: every rule still lives in `game.py`, so what you watch here is
exactly what an agent sees. Imported lazily by the CLI, because the engine and
the tests must keep working without pygame installed.
"""

from __future__ import annotations

import math
import threading
from collections import deque

import pygame

from .agents import Agent
from .game import Direction, GameState, SnakeGame

BG = (14, 17, 23)
PANEL = (22, 26, 36)
GRID = (30, 35, 48)
HEAD = (94, 234, 212)
BODY_NEAR = (45, 212, 191)
BODY_FAR = (13, 110, 104)
FOOD = (249, 115, 22)
FOOD_GLOW = (124, 45, 18)
TEXT = (226, 232, 240)
MUTED = (100, 116, 139)
DANGER = (248, 113, 113)

PAD = 18
HUD_TOP = 58
HUD_BOTTOM = 34
PANEL_W = 312
ACCENT = (56, 189, 248)
WARN = (251, 191, 36)
OK = (52, 211, 153)
GRID_TEXT = (71, 85, 105)

KEY_TO_DIRECTION = {
    pygame.K_UP: Direction.UP,
    pygame.K_w: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_a: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_d: Direction.RIGHT,
}


def _mix(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


class SnakeUI:
    """Fixed-timestep game loop with a free-running 60 fps redraw.

    The engine ticks on its own clock so gameplay speed does not depend on the
    frame rate, and the frames in between interpolate the head and tail to hide
    the grid steps.
    """

    def __init__(
        self,
        game: SnakeGame,
        *,
        agent: Agent | None = None,
        tick: float = 0.12,
        cell: int = 28,
    ) -> None:
        self.game = game
        self.agent = agent
        self.tick = tick
        self.cell = cell

        # Only an agent that keeps a decision log earns the side panel.
        self.log_panel = hasattr(agent, "log")
        self.board_px = (game.width * cell, game.height * cell)
        board_h = self.board_px[1] + PAD * 2 + HUD_TOP + HUD_BOTTOM
        window_h = max(board_h, 560 if self.log_panel else 0)
        self.top = PAD + (window_h - board_h) // 2
        size = (
            self.board_px[0] + PAD * 2 + (PANEL_W if self.log_panel else 0),
            window_h,
        )

        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Snake — Jev decision environment")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 34)
        self.font = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)

        self.origin = (PAD, self.top + HUD_TOP)
        self.pending: deque[Direction] = deque(maxlen=2)
        self.prev_state: GameState | None = None
        self.accumulator = 0.0
        self.since_step = 0.0
        self.paused = False
        self.elapsed = 0.0
        #: Set while a move is being computed off the render thread.
        self._thinking: dict | None = None

    # ---------------------------------------------------------------- loop

    def run(self) -> GameState:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            self.elapsed += dt
            running = self._handle_events()
            if running and not self.paused and not self.game.state.is_over:
                self.since_step += dt
                self.accumulator += dt
                if self.agent is None:
                    while self.accumulator >= self.tick:
                        self.accumulator -= self.tick
                        self._advance()
                else:
                    self._pump_agent()
            self._draw()
        pygame.quit()
        return self.game.state

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                return False
            if event.key == pygame.K_r:
                self._restart()
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key in KEY_TO_DIRECTION and self.agent is None:
                # Queued rather than applied now, so two quick presses inside
                # one tick both land instead of the first being overwritten.
                self.pending.append(KEY_TO_DIRECTION[event.key])
        return True

    def _restart(self) -> None:
        self.game.reset()
        self.pending.clear()
        self.prev_state = None
        self.accumulator = 0.0
        self.since_step = 0.0
        self._thinking = None
        self.paused = False

    def _pump_agent(self) -> None:
        """Step at the agent's pace without blocking the render loop.

        A Jev call takes hundreds of milliseconds, so asking for the move inline
        would freeze the window. The work runs on a daemon thread and the game
        advances on the frame the answer arrives.
        """

        if self._thinking is None:
            if self.accumulator < self.tick:
                return
            self._thinking = self._submit(self.game.state)
            return
        if not self._thinking["done"]:
            return
        move = self._thinking["move"]
        self._thinking = None
        self.accumulator = 0.0
        self.prev_state = self.game.state
        self.game.step(move)
        self.since_step = 0.0

    def _submit(self, state: GameState) -> dict:
        box: dict = {"done": False, "move": None}

        def work() -> None:
            try:
                box["move"] = self.agent.next_direction(state)
            finally:
                # The agent already logs its own failures; never wedge the loop.
                box["done"] = True

        threading.Thread(target=work, daemon=True).start()
        return box

    def _next_direction(self) -> Direction | None:
        if self.agent is not None:
            return self.agent.next_direction(self.game.state)
        current = self.game.state.direction
        while self.pending:
            move = self.pending.popleft()
            if move is not current.opposite:
                return move
        return None

    def _advance(self) -> None:
        self.prev_state = self.game.state
        self.game.step(self._next_direction())
        self.since_step = 0.0

    # -------------------------------------------------------------- drawing

    def _rect(self, cell_pos, inset: float = 0.0) -> pygame.Rect:
        x, y = cell_pos
        ox, oy = self.origin
        return pygame.Rect(
            ox + x * self.cell + inset,
            oy + y * self.cell + inset,
            self.cell - inset * 2,
            self.cell - inset * 2,
        )

    def _lerp_rect(self, start, end, t: float, inset: float) -> pygame.Rect:
        a, b = self._rect(start, inset), self._rect(end, inset)
        return pygame.Rect(
            round(a.x + (b.x - a.x) * t), round(a.y + (b.y - a.y) * t), a.w, a.h
        )

    def _draw(self) -> None:
        state = self.game.state
        # tick 0 means 'as fast as the agent answers', so there is nothing to
        # interpolate across.
        alpha = 1.0 if self.tick <= 0 else min(self.since_step / self.tick, 1.0)

        self.screen.fill(BG)
        self._draw_board_background()
        self._draw_tail(state, alpha)
        self._draw_body(state)
        self._draw_food(state)
        self._draw_head(state, alpha)
        self._draw_hud(state)
        if self.log_panel:
            self._draw_panel()
        if state.is_over or self.paused:
            self._draw_overlay(state)
        pygame.display.flip()

    def _draw_board_background(self) -> None:
        ox, oy = self.origin
        pygame.draw.rect(
            self.screen, PANEL, pygame.Rect(ox, oy, *self.board_px), border_radius=8
        )
        for x in range(1, self.game.width):
            px = ox + x * self.cell
            pygame.draw.line(self.screen, GRID, (px, oy), (px, oy + self.board_px[1]))
        for y in range(1, self.game.height):
            py = oy + y * self.cell
            pygame.draw.line(self.screen, GRID, (ox, py), (ox + self.board_px[0], py))

    def _draw_tail(self, state: GameState, alpha: float) -> None:
        """Slide the vacated cell forward onto the tail, so the body retracts.

        Shrinking it in place instead left a stray dot beside the tail.
        """

        prev = self.prev_state
        if prev is None or len(state.snake) > len(prev.snake):
            return
        gone = prev.snake[-1]
        if gone == state.snake[-1]:
            return
        rect = self._lerp_rect(gone, state.snake[-1], alpha, 2)
        pygame.draw.rect(self.screen, BODY_FAR, rect, border_radius=7)

    def _draw_body(self, state: GameState) -> None:
        segments = state.snake[1:]
        falloff = min(len(segments), 14)
        for index, cell in enumerate(segments):
            shade = _mix(BODY_NEAR, BODY_FAR, index / max(falloff - 1, 1))
            pygame.draw.rect(self.screen, shade, self._rect(cell, 2), border_radius=7)

    def _draw_head(self, state: GameState, alpha: float) -> None:
        prev = self.prev_state
        start = prev.head if prev else state.head
        rect = self._lerp_rect(start, state.head, alpha, 1)
        color = DANGER if state.is_over else HEAD
        pygame.draw.rect(self.screen, color, rect, border_radius=8)

        # Eyes, offset along the direction of travel so the head reads as facing.
        dx, dy = state.direction.delta
        eye = max(2, self.cell // 9)
        along, across = self.cell * 0.22, self.cell * 0.2
        cx, cy = rect.center
        for side in (-1, 1):
            ex = cx + dx * along - dy * across * side
            ey = cy + dy * along + dx * across * side
            pygame.draw.circle(self.screen, BG, (round(ex), round(ey)), eye)

    def _draw_food(self, state: GameState) -> None:
        if not state.food:
            return
        rect = self._rect(state.food)
        pulse = (math.sin(self.elapsed * 5) + 1) / 2
        radius = self.cell * 0.28 + self.cell * 0.05 * pulse
        pygame.draw.circle(self.screen, FOOD_GLOW, rect.center, round(radius * 1.6))
        pygame.draw.circle(self.screen, FOOD, rect.center, round(radius))

    def _draw_hud(self, state: GameState) -> None:
        right = self.origin[0] + self.board_px[0]
        top = self.top
        self.screen.blit(
            self.font_big.render(f"{state.score}", True, TEXT), (PAD, top - 2)
        )
        self.screen.blit(self.font_small.render("SCORE", True, MUTED), (PAD, top + 26))

        stats = f"length {len(state.snake)}    step {state.steps}"
        surf = self.font.render(stats, True, MUTED)
        self.screen.blit(surf, (right - surf.get_width(), top))

        label = self._agent_label()
        surf = self.font_small.render(label, True, MUTED)
        self.screen.blit(surf, (right - surf.get_width(), top + 24))

        footer = (
            "space pause    r restart    q quit"
            if self.agent
            else "arrows / wasd    space pause    r restart    q quit"
        )
        surf = self.font_small.render(footer, True, MUTED)
        self.screen.blit(surf, (PAD, self.origin[1] + self.board_px[1] + PAD - 4))

    def _agent_label(self) -> str:
        if self.agent is None:
            return "you are playing"
        stats = getattr(self.agent, "stats", None)
        if stats and stats.get("calls"):
            return f"{self.agent.name}    fallbacks {stats['fallbacks']}/{stats['calls']}"
        return self.agent.name

    # ---------------------------------------------------------------- panel

    def _panel_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.board_px[0] + PAD * 2,
            PAD,
            PANEL_W - PAD,
            self.screen.get_height() - PAD * 2,
        )

    def _draw_panel(self) -> None:
        rect = self._panel_rect()
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        x = rect.x + 14
        inner = rect.w - 28
        y = rect.y + 12

        y = self._panel_header(x, y, inner)
        y = self._panel_latest(x, y, inner)
        y = self._panel_bars(x, y, inner)
        y = self._panel_stats(x, y, inner)
        self._panel_history(x, y, inner, rect.bottom - 12)

    def _text(self, text, x, y, font=None, color=TEXT, right=None):
        font = font or self.font_small
        surf = font.render(text, True, color)
        self.screen.blit(surf, (right - surf.get_width() if right else x, y))
        return y + font.get_height()

    def _rule(self, x, y, width) -> int:
        pygame.draw.line(self.screen, GRID, (x, y), (x + width, y))
        return y + 10

    def _panel_header(self, x, y, width) -> int:
        self._text("JEV", x, y, self.font, ACCENT)
        model = getattr(self.agent, "model", None) or "waiting for first reply"
        y = self._text(model, x, y + 3, self.font_small, MUTED, right=x + width)
        return self._rule(x, y + 8, width)

    def _panel_latest(self, x, y, width) -> int:
        if self._thinking is not None:
            dots = "." * (1 + int(self.elapsed * 3) % 3)
            return self._headline(x, y, width, f"thinking{dots}", MUTED, "")

        last = self._last_decision()
        if last is None:
            return self._headline(x, y, width, "no calls yet", MUTED, "")

        if last.accepted:
            return self._headline(x, y, width, last.applied, OK, "", last)
        note = f"answered {last.answer}, {last.note}" if last.answer else last.note
        return self._headline(x, y, width, last.applied, WARN, note, last)

    def _headline(self, x, y, width, headline, colour, note, last=None) -> int:
        self._text(headline, x, y, self.font_big, colour)
        if last is not None and last.confidence is not None:
            self._text(
                f"confidence {last.confidence:.2f}",
                x,
                y + 8,
                self.font_small,
                MUTED,
                right=x + width,
            )
        y += self.font_big.get_height() + 2
        if note:
            y = self._text(note, x, y, self.font_small, WARN)
        return self._rule(x, y + 8, width)

    def _panel_bars(self, x, y, width) -> int:
        last = self._last_decision()
        probabilities = last.probabilities if last else {}
        safe = set(last.safe) if last else set()
        for name in ("UP", "DOWN", "LEFT", "RIGHT"):
            value = probabilities.get(name, 0.0)
            unsafe = bool(last) and name not in safe
            label_colour = DANGER if unsafe else TEXT
            self._text(f"x {name}" if unsafe else name, x, y, self.font_small, label_colour)

            bar = pygame.Rect(x + 62, y + 2, width - 106, 10)
            pygame.draw.rect(self.screen, GRID, bar, border_radius=5)
            filled = round(bar.w * value)
            if filled >= 1:
                colour = DANGER if unsafe else (ACCENT if last and name == last.answer else OK)
                pygame.draw.rect(
                    self.screen, colour, pygame.Rect(bar.x, bar.y, filled, bar.h),
                    border_radius=5,
                )

            self._text(
                f"{value:.0%}" if probabilities else "-",
                x,
                y,
                self.font_small,
                label_colour,
                right=x + width,
            )
            y += 20
        return self._rule(x, y + 2, width)

    def _panel_stats(self, x, y, width) -> int:
        stats = getattr(self.agent, "stats", {})
        rows = [
            ("calls", str(stats.get("calls", 0))),
            (
                "fallbacks",
                f"{stats.get('fallbacks', 0)}  ({stats.get('errors', 0)} errors)",
            ),
            ("mean latency", f"{getattr(self.agent, 'mean_latency_ms', 0):.0f} ms"),
            (
                "tokens",
                f"{stats.get('input_tokens', 0)} in / {stats.get('output_tokens', 0)} out",
            ),
        ]
        for label, value in rows:
            self._text(label, x, y, self.font_small, MUTED)
            y = self._text(value, x, y, self.font_small, TEXT, right=x + width)
            y += 2
        return self._rule(x, y + 6, width)

    def _panel_history(self, x, y, width, bottom) -> None:
        self._text("RECENT", x, y, self.font_small, MUTED)
        y += 20
        log = getattr(self.agent, "log", None)
        if not log:
            return
        line_h = self.font_small.get_height() + 3
        room = max(0, int((bottom - y) // line_h))
        for entry in list(log)[-room:][::-1]:
            colour = MUTED if entry.accepted else WARN
            detail = entry.applied if entry.accepted else f"{entry.applied}  {entry.note}"
            self._text(f"{entry.step:>4}", x, y, self.font_small, GRID_TEXT)
            self._text(detail, x + 34, y, self.font_small, colour)
            if entry.confidence is not None:
                self._text(
                    f"{entry.confidence:.2f}",
                    x,
                    y,
                    self.font_small,
                    MUTED,
                    right=x + width,
                )
            y += line_h

    def _last_decision(self):
        log = getattr(self.agent, "log", None)
        return log[-1] if log else None

    def _draw_overlay(self, state: GameState) -> None:
        veil = pygame.Surface(self.board_px, pygame.SRCALPHA)
        veil.fill((10, 12, 17, 205))
        self.screen.blit(veil, self.origin)

        if state.is_over:
            reasons = {
                "hit_wall": "ran into the wall",
                "hit_self": "bit itself",
                "starved": "starved",
            }
            title = "you filled the board" if state.status == "won" else "game over"
            subtitle = reasons.get(state.reason or "", state.status)
            lines = [(title, self.font_big, TEXT), (subtitle, self.font, MUTED)]
            lines.append((f"score {state.score}", self.font, TEXT))
            lines.append(("press r to play again", self.font_small, MUTED))
        else:
            lines = [("paused", self.font_big, TEXT), ("press space", self.font_small, MUTED)]

        total = sum(f.get_height() + 8 for _, f, _ in lines)
        y = self.origin[1] + (self.board_px[1] - total) // 2
        for text, font, color in lines:
            surf = font.render(text, True, color)
            self.screen.blit(
                surf, (self.origin[0] + (self.board_px[0] - surf.get_width()) // 2, y)
            )
            y += font.get_height() + 8


def run_ui(game: SnakeGame, **kwargs) -> GameState:
    return SnakeUI(game, **kwargs).run()
