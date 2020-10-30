from concurrent.futures import ThreadPoolExecutor
import threading
import pygame
import time
import sys
import traceback
from sevn import Game, State, Score, Board
from agents.random_agent import RandomAgent
from agents.human import Human
from agents.mcts_agent import MinimaxMCTS
from agents.uct_agent import UCTAgent

class ThreadPoolExecutorTimedStackTraced(ThreadPoolExecutor):
    """
    A ThreadPoolExecutor which displays a traceback when an exception is thrown during thread execution.
    Taken from: https://stackoverflow.com/a/24457608.
    """

    def submit(self, fn, *args, **kwargs):
        """Submits the wrapped function instead of `fn`"""

        self.start_time = time.time()

        return super(ThreadPoolExecutorTimedStackTraced, self).submit(
            self._function_wrapper, fn, *args, **kwargs)

    def _function_wrapper(self, fn, *args, **kwargs):
        """Wraps `fn` in order to preserve the traceback of any kind of
        raised exception

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
        self.signal = threading.Event() # Signal to unblock the human agent class, either when the user submits a move or the program is terminated.
        self.selected = {} # All tiles selected by the user.
        self.terminate = False # Flag telling the human agent class whether to terminate or not once the signal is set.

class Animations:

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

default_colors = {
    0: (237, 184, 121),
    1: (224, 127, 133),
    2: (128, 90, 91),
    3: (127, 55, 115),
    4: (240, 222, 196),
    5: (63, 61, 55),
    6: (136, 159, 175)
}

if __name__ == "__main__":
    base = 5
    # game = Game(base)
    game = Game.from_str("1/aaaaa/eccda.bdcbb.bacba.ddecd.eaeea")

    user_input = UserInput()

    game.reset_search_game()
    player2 = UCTAgent(game.search_game)
    # player1 = MinimaxMCTS(game.search_game)
    player1 = Human(game.search_game, user_input)

    name_width = max(len(player1.name), len(player2.name)) + 4
    result_width = 3 + 2*base + base**2

    print(game.state)
    print(f"{' Name':<{name_width}}{'Time':<10}{'Confidence':<11}Result")
    print("-"*(name_width + 21 + result_width))
    print(" "*(name_width + 21) + str(game.state))

    next_player = player1 if game.state.next_go == 1 else player2

    agent_executor = ThreadPoolExecutorTimedStackTraced()
    agent_future = agent_executor.submit(next_player.select_move)

    pygame.init()

    small_font = pygame.font.SysFont("Bahnschrift", 20)
    big_font = pygame.font.SysFont("Bahnschrift", 50)

    animations = Animations()
    x_size, y_size = 600, 600
    screen = pygame.display.set_mode((x_size, y_size), pygame.RESIZABLE)
    pygame.display.set_caption("Pygame Template")

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
                pos = screen_to_board(mx, my)
                tile = (int(pos[1]/tile_size), int(pos[0]/tile_size))
                user_input.selected[tile] = not user_input.selected.get(tile, False)
            
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    user_input.signal.set()
        
        # ------------------------------------ Move making ------------------------------------
        if animations.done() and game.state.outcome == 0 and agent_future.done():

            user_input.selected.clear()
            move, time_taken = agent_future.result()
            for tile in move:
                animations.animate(tile, game.get_at(tile))
            
            game.make_move(move)
            game.reset_search_game()

            confidence_str = " "*11 if not next_player.confidence else f"{next_player.confidence+'  ':>11}"

            print(f"{' '+next_player.name:<{name_width}}{'%.2f'%time_taken+'s  ':>10}{confidence_str}{str(game.state)}")
            next_player = player1 if game.state.next_go == 1 else player2

            if game.state.outcome == 0:
                agent_future = agent_executor.submit(next_player.select_move)
                start_time = time.time()

        # ------------------------------------- Rendering -------------------------------------
        screen.fill([177, 161, 179])

        margin_size = 20
        
        score_width = min(int(2.5*y_size/3), x_size - 2*margin_size)
        
        grid_width = int(score_width*0.6) - int(score_width*0.6)%(2*base + 1) + 1
        cell_size = int((grid_width - 1)/(2*base + 1))
        grid_height = int(base*cell_size + 1)

        score_height = int(grid_height + 50)
        board_size = min(y_size - score_height - 3*margin_size, x_size - 2*margin_size)

        score_surf = pygame.Surface((score_width, score_height), pygame.SRCALPHA, 32)
        board_surf = pygame.Surface((board_size, board_size), pygame.SRCALPHA, 32)

        def board_to_screen(x, y):
            return (x + int((x_size - board_size)/2), y + margin_size*2 + score_height)

        def screen_to_board(x, y):
            return (x - int((x_size - board_size)/2), y - margin_size*2 - score_height)

        if animations.done() and game.state.outcome != 0: # if someone has won
            
            winner = player1 if game.state.outcome == 1 else player2
            
            win_label = big_font.render(winner.name + " wins!", 1, [255,255,255])
            win_label_rect = win_label.get_rect()
            win_label_rect.center = (x_size/2, y_size/2)

            screen.blit(win_label, win_label_rect)
        else: # else draw the board
            for row in range(base):
                for col in range(base):
                    tile_size = int(board_size/base)
                    if user_input.selected.get((row, col)) and (row, col) in game.get_takable():
                        pygame.draw.rect(board_surf, [255,255,255], (tile_size*col + 2, tile_size*row + 2, tile_size-4, tile_size-4))
                        pygame.draw.rect(board_surf, [0,0,0], (tile_size*col + 4, tile_size*row + 4, tile_size-8, tile_size-8))

                    tile = game.get_at(row, col)
                    if tile >= 0: 
                        pygame.draw.rect(board_surf, default_colors[tile], (tile_size*col + 5, tile_size*row + 5, tile_size-10, tile_size-10))
            
            animations.step()
            for pos, color, lerp in animations.tiles:
                pygame.draw.rect(board_surf, default_colors[color], (tile_size*pos[1] + 5 + (tile_size/2 - 5)*lerp, tile_size*pos[0] + 5 + (tile_size/2 - 5)*lerp, (tile_size-10)*(1 - lerp), (tile_size-10)*(1 - lerp)))
        
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
            pygame.draw.line(score_surf, unselected_col, (0, 30), (int((score_width - cell_size)/2), 30), 2)
            pygame.draw.line(score_surf, selected_col, (0, 30), (int(progress*(score_width - cell_size)/2), 30), 2)
        else:
            pygame.draw.line(score_surf, p1_color, (0, 30), (int((score_width - cell_size)/2), 30), 2)

        if progress and next_player == player2:
            pygame.draw.line(score_surf, unselected_col, (int((score_width + cell_size)/2), 30), (score_width, 30), 2)
            pygame.draw.line(score_surf, selected_col, (int((score_width + cell_size)/2 + (1-progress)*(score_width - cell_size)/2), 30), (score_width, 30), 2)
        else:
            pygame.draw.line(score_surf, p2_color, (int((score_width + cell_size)/2), 30), (score_width, 30), 2)

        # Draw the score grid
        grid_surf = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA, 32)
        for row in range(base + 1):
            pygame.draw.line(grid_surf, [200,200,200], (0, row*cell_size), (cell_size*base, row*cell_size))
            pygame.draw.line(grid_surf, [200,200,200], (cell_size*(base + 1), row*cell_size), (grid_width, row*cell_size))
        for col in range(2*base + 2):
            pygame.draw.line(grid_surf, [200,200,200], (col*cell_size, 0), (col*cell_size, grid_height))
        for i in range(base):
            pygame.draw.rect(grid_surf, [255,255,255], ((base - game.get_score(i))*cell_size, i*cell_size, cell_size + 1, cell_size + 1))
            pygame.draw.rect(grid_surf, default_colors[i], ((base - game.get_score(i))*cell_size + 1, i*cell_size + 1, cell_size - 1, cell_size - 1))
        
        score_surf.blit(grid_surf, (int((score_width - grid_width)/2), score_height - grid_height))

        # Draw the score numbers
        p1_score, p2_score = game.state.score.get_score_pair()

        p1_score_label = big_font.render(str(p1_score), 1, [255,255,255])
        p2_score_label = big_font.render(str(p2_score), 1, [255,255,255])

        p1_score_label_rect = p1_score_label.get_rect()
        p1_score_label_rect.topleft = (0, score_height - grid_height)

        p2_score_label_rect = p2_score_label.get_rect()
        p2_score_label_rect.topright = (score_width, score_height - grid_height)

        score_surf.blit(p1_score_label, p1_score_label_rect)
        score_surf.blit(p2_score_label, p2_score_label_rect)

        # Draw the confidences
        if player1.confidence != None:
            confidence_label = small_font.render(str(player1.confidence), 1, [255,255,255])

            confidence_label_rect = confidence_label.get_rect()
            confidence_label_rect.bottomleft = (0, score_height)
        
            score_surf.blit(confidence_label, confidence_label_rect)
        
        if player2.confidence != None:
            confidence_label = small_font.render(str(player2.confidence), 1, [255,255,255])

            confidence_label_rect = confidence_label.get_rect()
            confidence_label_rect.bottomright = (score_width, score_height)
        
            score_surf.blit(confidence_label, confidence_label_rect)
        
        # Blit the surfaces on the screen
        screen.blit(score_surf, (int((x_size - score_width)/2), margin_size))
        screen.blit(board_surf, board_to_screen(0,0))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    user_input.terminate = True
    user_input.signal.set()
    agent_future.cancel()