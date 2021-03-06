from concurrent.futures import ThreadPoolExecutor
import threading
import torch
import pygame
import time
import sys
import traceback
import utils
from agents.eta_zero import EtaZero, EtaZeroVisualiser
from agents.human import Human
from agents.mcts_agent import MinimaxMCTS
from agents.network_agent import RawNetwork
from agents.random_agent import RandomAgent
from agents.uct_agent import UCTAgent
from game.renderer import Renderer
from game.sevn import Board, Game, Score, State
from ios_screen_capture import screen_parser
from networks.graph_networks import DGLValueWinNetwork, PolicyValRGCN
from networks.dummy_networks import DummyPVNetwork


# from screen_parsing import simple_plotter


class ThreadPoolExecutorTimedStackTraced(ThreadPoolExecutor):
    """
    A ThreadPoolExecutor with two modifications:
        - returns a pair of (return value, time taken)
        - displays a traceback when an exception is thrown during thread execution
    Modified from: https://stackoverflow.com/a/24457608.
    """

    def submit(self, fn, *args, **kwargs):
        """
        Submits the wrapped function instead of `fn`.
        Starts the timer.
        """
        self.start_time = time.time()

        return super(ThreadPoolExecutorTimedStackTraced, self).submit(
            self._function_wrapper, fn, *args, **kwargs
        )

    def _function_wrapper(self, fn, *args, **kwargs):
        """
        Wraps `fn` in order to preserve the traceback of any kind of raised exception.
        Returns the value returned by the function and the time it took to execute.
        """
        try:
            return (fn(*args, **kwargs), time.time() - self.start_time)
        except Exception:
            raise sys.exc_info()[0](traceback.format_exc())


class UserInput:
    """
    A structure to store the user input necessary for a human agent class to interpret the chosen move of the player.
    """

    def __init__(self):
        # Signal to unblock the human agent class, either when the user submits a move or the program is terminated.
        self.signal = threading.Event()
        self.selected = {}  # All tiles selected by the user.
        self.selected_col = None  # The colour of the selected tile(s)
        # Flag telling the human agent class whether to terminate or not once the signal is set.
        self.terminate = False


class Animations:
    """
    A structure to keep track of tiles which are animating.
    """

    def __init__(self):
        self.tiles = []

    def animate(self, position, color):
        self.tiles.append((position, color, 0))

    def step(self):
        self.tiles = [
            (position, color, lerp + 0.015)
            for position, color, lerp in self.tiles
            if lerp + 0.015 < 1
        ]

    def done(self):
        return len(self.tiles) == 0


default_back_col = (95, 46, 95), (226, 171, 152)
user_input = UserInput()


def play_game(
    game,
    player1,
    player2,
    tile_colors=Renderer.default_colors,
    top_col=default_back_col[0],
    bot_col=default_back_col[1],
):

    base = game.base

    game.reset_search_game()
    player1.set_game(game.search_game)
    player2.set_game(game.search_game)

    name_width = max(len(player1.name), len(player2.name)) + 4
    result_width = 3 + 2 * base + base ** 2

    def pprint_row(name, time, confidence, playouts, result):
        print(
            f"{' '+name:<{name_width}}{time:<10}{confidence:<11}{playouts:<11}{result}"
        )

    pprint_row("Name", "Time", "Confidence", "Playouts", "Result")
    print("-" * (name_width + 32 + result_width))
    print(" " * (name_width + 32) + str(game.state))

    next_player = player1 if game.state.next_go == 1 else player2

    state_list = [game.state]

    agent_executor = ThreadPoolExecutorTimedStackTraced()

    pygame.init()

    small_font = pygame.font.SysFont("Bahnschrift", 20)
    big_font = pygame.font.SysFont("Bahnschrift", 50)
    board_surf = None

    animations = Animations()
    x_size, y_size = 600, 600
    screen = pygame.display.set_mode((x_size, y_size), pygame.RESIZABLE)
    pygame.display.set_caption("Sevn")

    agent_future = agent_executor.submit(next_player.select_move)

    clock = pygame.time.Clock()
    anim_done = True
    done = False
    while not done:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            if event.type == pygame.VIDEORESIZE:
                x_size = max(600, event.w)
                y_size = max(600, event.h)
                screen = pygame.display.set_mode((x_size, y_size), pygame.RESIZABLE)

            if event.type == pygame.MOUSEBUTTONUP:
                if board_surf:
                    pos = screen_to_board(mx, my)
                    tile_size = board_surf.get_size()[0] // base
                    tile = (int(pos[1] / tile_size), int(pos[0] / tile_size))

                    if tile in game.get_takable():
                        if game.get_at(tile) != user_input.selected_col:
                            user_input.selected.clear()

                        user_input.selected_col = game.get_at(tile)
                        user_input.selected[tile] = not user_input.selected.get(
                            tile, False
                        )

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    user_input.signal.set()
                if event.key == pygame.K_v:
                    # if game.over():
                    eta = None
                    if isinstance(player1, EtaZero):
                        eta = player1
                    elif isinstance(player2, EtaZero):
                        eta = player2

                    if eta is not None:
                        EtaZeroVisualiser(eta, state_list)
                        done = True

        # ------------------------------------ Move making ------------------------------------
        if animations.done() and game.state.outcome == 0 and agent_future.done():

            user_input.selected.clear()
            user_input.selected_col = None
            move, time_taken = agent_future.result()
            for tile in move:
                animations.animate(tile, game.get_at(tile))

            game.make_move(move)
            game.reset_search_game()
            state_list.append(game.state)

            confidence_str = (
                " " * 11
                if not next_player.confidence
                else f"{next_player.confidence+'  ':>11}"
            )
            playouts_str = (
                " " * 11
                if not next_player.playouts_played
                else f"{str(next_player.playouts_played)+'/'+str(len(game.get_moves()))+'  ':>11}"
            )

            pprint_row(
                name=next_player.name,
                time=f"{'%.2f'%time_taken+'s '}",
                confidence=confidence_str,
                playouts=playouts_str,
                result=str(game.state),
            )

            next_player = player1 if game.state.next_go == 1 else player2

            if not game.over():
                agent_future = agent_executor.submit(next_player.select_move)

            else:
                print(game.state.get_game_str())

        # ------------------------------------- Rendering -------------------------------------
        screen.fill([177, 161, 179])

        margin_size = 20

        score_width = min(int(2.5 * y_size / 3), x_size - 2 * margin_size)

        grid_width = (
            int(score_width * 0.6) - int(score_width * 0.6) % (2 * base + 1) + 1
        )
        cell_size = int((grid_width - 1) / (2 * base + 1))
        grid_height = int(base * cell_size + 1)

        score_height = int(grid_height + 50)
        board_size = min(
            y_size - score_height - 3 * margin_size, x_size - 2 * margin_size
        )

        score_surf = pygame.Surface((score_width, score_height), pygame.SRCALPHA, 32)
        board_surf = pygame.Surface((board_size, board_size), pygame.SRCALPHA, 32)

        def board_to_screen(x, y):
            return (
                x + int((x_size - board_size) / 2),
                y + margin_size * 2 + score_height,
            )

        def screen_to_board(x, y):
            return (
                x - int((x_size - board_size) / 2),
                y - margin_size * 2 - score_height,
            )

        background = pygame.Surface((x_size, y_size))
        for y in range(y_size):
            col_a, col_b = (
                default_back_col
                if y + margin_size / 2 - 3 < board_to_screen(0, 0)[1]
                else (top_col, bot_col)
            )
            lerp_col = [a + (b - a) * y / y_size for a, b in zip(col_a, col_b)]
            pygame.draw.line(background, lerp_col, (0, y), (x_size - 1, y))

        screen.blit(background, (0, 0))

        if animations.done() and game.state.outcome != 0:  # if someone has won

            winner = player1 if game.state.outcome == 1 else player2

            win_label = big_font.render(winner.name + " wins!", 1, [255, 255, 255])
            win_label_rect = win_label.get_rect()
            win_label_rect.center = (x_size / 2, y_size / 2)

            screen.blit(win_label, win_label_rect)
        else:  # else draw the board
            animations.step()
            Renderer.draw_board(
                board_surf,
                game.state.board,
                user_input.selected,
                animations.tiles,
                colors=tile_colors,
            )

        # Draw the player name labels
        unselected_col = (196, 187, 173)
        selected_col = (242, 193, 44)

        p1_color = selected_col if game.state.next_go == 1 else unselected_col
        p2_color = unselected_col if game.state.next_go == 1 else selected_col

        p1_label = small_font.render(player1.name, 1, p1_color)
        p2_label = small_font.render(player2.name, 1, p2_color)

        p1_label_rect = p1_label.get_rect()
        p1_label_rect.topleft = (0, 0)

        p2_label_rect = p2_label.get_rect()
        p2_label_rect.topright = (score_width, 0)

        score_surf.blit(p1_label, p1_label_rect)
        score_surf.blit(p2_label, p2_label_rect)

        progress = next_player.get_progress()

        if progress and next_player == player1:
            pygame.draw.line(
                score_surf,
                unselected_col,
                (0, 30),
                (int((score_width - cell_size) / 2), 30),
                2,
            )
            pygame.draw.line(
                score_surf,
                selected_col,
                (0, 30),
                (int(progress * (score_width - cell_size) / 2), 30),
                2,
            )
        else:
            pygame.draw.line(
                score_surf,
                p1_color,
                (0, 30),
                (int((score_width - cell_size) / 2), 30),
                2,
            )

        if progress and next_player == player2:
            pygame.draw.line(
                score_surf,
                unselected_col,
                (int((score_width + cell_size) / 2), 30),
                (score_width, 30),
                2,
            )
            pygame.draw.line(
                score_surf,
                selected_col,
                (
                    int(
                        (score_width + cell_size) / 2
                        + (1 - progress) * (score_width - cell_size) / 2
                    ),
                    30,
                ),
                (score_width, 30),
                2,
            )
        else:
            pygame.draw.line(
                score_surf,
                p2_color,
                (int((score_width + cell_size) / 2), 30),
                (score_width, 30),
                2,
            )

        # Draw the score grid
        grid_surf = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA, 32)
        Renderer.draw_score_grid(grid_surf, game.state.score, colors=tile_colors)
        score_surf.blit(
            grid_surf, (int((score_width - grid_width) / 2), score_height - grid_height)
        )

        # Draw the score numbers
        p1_score, p2_score = game.state.score.get_score_pair()

        p1_score_label = big_font.render(str(p1_score), 1, [255, 255, 255])
        p2_score_label = big_font.render(str(p2_score), 1, [255, 255, 255])

        p1_score_label_rect = p1_score_label.get_rect()
        p1_score_label_rect.topleft = (0, score_height - grid_height)

        p2_score_label_rect = p2_score_label.get_rect()
        p2_score_label_rect.topright = (score_width, score_height - grid_height)

        score_surf.blit(p1_score_label, p1_score_label_rect)
        score_surf.blit(p2_score_label, p2_score_label_rect)

        # Draw the confidences
        if player1.confidence != None:
            confidence_label = small_font.render(
                str(player1.confidence), 1, [255, 255, 255]
            )

            confidence_label_rect = confidence_label.get_rect()
            confidence_label_rect.bottomleft = (0, score_height)

            score_surf.blit(confidence_label, confidence_label_rect)

        if player2.confidence != None:
            confidence_label = small_font.render(
                str(player2.confidence), 1, [255, 255, 255]
            )

            confidence_label_rect = confidence_label.get_rect()
            confidence_label_rect.bottomright = (score_width, score_height)

            score_surf.blit(confidence_label, confidence_label_rect)

        # Blit the surfaces on the screen
        screen.blit(score_surf, (int((x_size - score_width) / 2), margin_size))
        screen.blit(board_surf, board_to_screen(0, 0))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    user_input.terminate = True
    user_input.signal.set()
    agent_future.cancel()


if __name__ == "__main__":
    """Random game state
    game = Game(base)
    """

    """ Game state from a board
    game = Game(
        base,
        State(
            Board(base, simple_plotter.get_board()),
            Score(base),
            1
        )
    ) """

    """ Game state from string representation
    game = Game.from_str("1/aaa/acb.bca.cba")
    """

    """ Game state from iOS App Sevn
    state, *colors = screen_parser.get_starting_state()
    game = Game(state=state)
    play_game(Game(state=state), player1, player2, *colors)
    """

    game = Game(7)

    player1 = Human(user_input)
    player2 = EtaZero(utils.load_net(80, section="Attempt7"), samples_per_move=50)

    play_game(game, player1, player2)
