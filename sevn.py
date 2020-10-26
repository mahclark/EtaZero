from random import shuffle
import pygame
import time

class Game:
    """
    Defines the game logic.
    """

    def __init__(self, base=7):
        if base % 2 == 0 or base < 5:
            raise Exception("Invalid base number. Must be odd and > 3.")

        self.base = base

        self.state = State(
            board=Board(base),
            score=Score(base),
            next_go=1, # Player 1 or -1 to play?
            outcome=0, # Who has won? 0 if incomplete
        )

    def get_moves(self):
        return self.state.board.get_moves()
    
    def get_takable(self):
        return self.state.board.get_takable()
    
    def get_at(self, a, b=None):
        return self.state.board.get_at(a, b)
    
    def get_score(self, i):
        return self.state.score.score[i]
    
    def make_move(self, move):
        if move in self.state.children:
            self.state = self.state.children[move]
            return

        board = self.state.board.make_move(move)
        score = self.state.score.make_move(move, self.get_at(next(iter(move))), self.state.next_go)

        if score.get_player_with_all() != 0:
            outcome = score.get_player_with_all()
        elif board.empty:
            p1score, p2score = score.get_score_pair()
            if p1score == p2score:
                raise Exception("Internal error: player scores equal when board is empty.")
            if p1score > p2score:
                outcome = 1
            else:
                outcome = -1
        else:
            outcome = 0
        
        self.state = State(
            board=board,
            score=score,
            next_go=-self.state.next_go,
            outcome=outcome,
            parent=self.state
        )

        self.state.parent.children[move] = self.state
    
    def undo_move(self):
        self.state = self.state.parent

class Board:
    def __init__(self, base, board=None):
        self.base = base

        if board == None:
            tiles = [i % base for i in range(base**2)]
            shuffle(tiles)
            self.board = [[tiles.pop() for _ in range(base)] for _ in range(base)]
        else:
            self.board = board
        
        self.empty = sum([sum(row) for row in self.board]) == -base**2
        
        self.takable = [
            (row, col)
            for row in range(self.base)
            for col in range(self.base)
            if self.get_at(row, col) >= 0
            and (self.get_at(row-1, col) == -1 or self.get_at(row+1, col) == -1)
            and (self.get_at(row, col-1) == -1 or self.get_at(row, col+1) == -1)
        ]

        takable_colors = {
            i: [
                tile
                for tile in self.get_takable()
                if self.get_at(tile) == i
            ]
            for i in range(self.base)
        }

        self.moves = {
            frozenset(comb)
            for tiles in takable_colors.values()
            for comb in self._all_combs(tiles)
        }
    
    @staticmethod
    def _all_combs(a):
        combs = []
        for b in range(1, 2**len(a)):
            combs.append({
                a[i]
                for i in range(len(a))
                if (b >> i) & 1 == 1
            })
        
        return combs
    
    def get_at(self, a, b=None):
        if b == None:
            row, col = a
        else:
            row, col = (a, b)
        
        if row < 0 or col < 0 or row >= self.base or col >= self.base:
            return -1
        return self.board[row][col]
    
    def make_move(self, move):
        if (len(move) == 0):
            raise Exception("Invalid move: must select a tile.")
        color = self.get_at(next(iter(move)))
        for tile in move:
            if self.get_at(tile) == -1:
                raise Exception("Invalid move: tile at {} already taken.".format(tile))
            if self.get_at(tile) != color:
                raise Exception("Invalid move: cannot take multiple colors.")
        
        return Board(
            self.base,
            [
                [-1
                if (row, col) in move
                else self.get_at(row, col)
                for col in range(self.base)]
                for row in range(self.base)
            ]
        )
    
    def is_empty(self):
        return self.empty
    
    def get_takable(self):
        return self.takable

    def get_moves(self):
        return self.moves

class Score:
    def __init__(self, base, score=None):
        self.base = base

        if score == None:
            self.score = [0 for _ in range(base)]
        else:
            self.score = score
        
        if sum([abs(x) > self.base for x in self.score]) > 0:
            raise Exception("Internal error: score exceeded base number {}.".format(self.base))
        
        if sum([abs(x) == self.base for x in self.score]) > 1:
            raise Exception("Internal error: multiple colors have been entirely captured by a player. Score:" + str(self.score))

        self.player_with_all = sum([int(x/self.base) for x in self.score])
        self.score_pair = (
            sum([x > 0 for x in self.score]),
            sum([x < 0 for x in self.score])
        )
        
    def make_move(self, move, color, next_go):
        diff = next_go*len(move)

        return Score(
            self.base,
            [
                self.score[i] + diff
                if i == color
                else self.score[i]
                for i in range(self.base)
            ]
        )
    
    def get_player_with_all(self):
        return self.player_with_all
    
    def get_score_pair(self):
        return self.score_pair

class State:
    def __init__(self, board, score, next_go, outcome, parent=None):
        self.board = board
        self.score = score
        self.next_go = next_go
        self.outcome = outcome
        self.parent = parent
        self.children = {}

if __name__ == "__main__":
    game = Game(5)
    game.make_move(next(iter(game.get_moves())))
    game.make_move(next(iter(game.get_moves())))
    game.make_move(next(iter(game.get_moves())))
    game.make_move(next(iter(game.get_moves())))
    game.make_move(next(iter(game.get_moves())))
    game.make_move(next(iter(game.get_moves())))

    for row in range(game.base):
        for col in range(game.base):
            tile = game.state.board.board[row][col]
            if tile == -1:
                print("- ",end="")
            elif (row, col) in game.state.board.get_takable():
                print("O ",end="")
            else:
                print("X ",end="")
        print()