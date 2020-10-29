from random import shuffle
import pygame
import time

class Game:
    """
    Defines the game logic.
    """

    def __init__(self, base=7, state=None):
        if base % 2 == 0 or base < 5:
            raise Exception("Invalid base number. Must be odd and > 3.")

        self.base = base

        if state == None:
            self.state = State(
                board=Board(base),
                score=Score(base),
                next_go=1 # Player 1 or -1 to play?
            )
        else:
            self.state = state
        
        self.search_game = None # Game object used by agents to search the state tree

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
        
        self.state = State(
            board=board,
            score=score,
            next_go=-self.state.next_go,
            parent=self.state
        )

        self.state.parent.children[move] = self.state
    
    def undo_move(self):
        self.state = self.state.parent
    
    def reset_search_game(self):
        if not self.search_game:
            self.search_game = Game(self.base, self.state)
        self.search_game.state = self.state
    
    @staticmethod
    def from_str(s):
        parts = s.split('/')
        if len(parts) != 3:
            raise Exception("Must have 3 parts separated by '/'.")

        if parts[0] not in ["1", "2"]:
            raise Exception("Expected first part to be 1 or 2; got {}.".format(parts[0]))

        next_go = int(parts[0])
        score = Score.from_str(parts[1])
        board = Board.from_str(parts[2])

        if score.base != board.base:
            raise Exception("Score and board must have the same base number.")

        return Game(
            score.base,
            State(
                board,
                score,
                next_go
            )
        )

class Board:
    def __init__(self, base, board=None):
        self.base = base

        if board == None:
            tiles = [i % base for i in range(base**2)]
            shuffle(tiles)
            self.board = [[tiles.pop() for _ in range(base)] for _ in range(base)]
        else:
            self.board = board
        
        self.board = tuple([tuple(row) for row in self.board])
        
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

        self.hash_val = hash(self.board)
    
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
    
    @staticmethod
    def validate(board):
        base = len(board)
        if base%2 != 1 or base < 5 or base > 25:
            raise Exception("Base number must be odd and between 4 and 26.")

        if sum([len(row) != base for row in board]) > 0:
            raise Exception("All rows must have length as the base number.")

        if sum([board[row][col] < -1 or board[row][col] >= base for row in range(base) for col in range(base)]) > 0:
            raise Exception("Tiles must be >= -1 and <= base number.")
    
    @staticmethod
    def from_str(s):
        lines = s.split(".")
        base = len(lines)

        board = []
        for line in lines:
            row = []
            for c in line:
                if c.isdigit():
                    for _ in range(int(c)):
                        row.append(-1)
                else:
                    row.append(ord(c) - ord('a'))
            board.append(row)
        
        Board.validate(board)
        
        return Board(base, board)

    def __hash__(self):
        return self.hash_val
    
    def __eq__(self, other):
        return other and self.board == other.board
    
    def __ne__(self, other):
        return not self.__eq__(other)

class Score:
    def __init__(self, base, score=None):
        self.base = base

        if score == None:
            self.score = tuple([0 for _ in range(base)])
        else:
            self.score = tuple(score)
        
        if sum([abs(x) > self.base for x in self.score]) > 0:
            raise Exception("Internal error: score exceeded base number {}.".format(self.base))
        
        if sum([abs(x) == self.base for x in self.score]) > 1:
            raise Exception("Internal error: multiple colors have been entirely captured by a player. Score:" + str(self.score))

        self.player_with_all = sum([int(x/self.base) for x in self.score])
        self.score_pair = (
            sum([x > 0 for x in self.score]),
            sum([x < 0 for x in self.score])
        )

        self.hash_val = hash(self.score)
        
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
    
    @staticmethod
    def validate(score):
        base = len(score)
        if base%2 != 1 or base < 5 or base > 25:
            raise Exception("Base number must be odd and between 4 and 26.")

        if sum([abs(x) > base for x in score]) > 0:
            raise Exception("Score cannot have values > base number.")

    @staticmethod
    def from_str(s):
        score = []

        neg = False
        for c in s:
            if c == '-':
                if not neg:
                    neg = True
                else:
                    raise Exception("Invalid string: found double '-' in score.")
            elif not neg:
                score.append(ord(c) - ord('a'))
            else:
                score.append(ord('a') - ord(c))
                neg = False
        
        Score.validate(score)
        
        return Score(len(score), score)
    
    def __hash__(self):
        return self.hash_val
    
    def __eq__(self, other):
        return other and self.score == other.score
    
    def __ne__(self, other):
        return not self.__eq__(other)

class State:
    def __init__(self, board, score, next_go, parent=None):
        self.board = board
        self.score = score
        self.next_go = next_go
        self.parent = parent
        self.children = {}

        if score.get_player_with_all() != 0:
            self.outcome = score.get_player_with_all()
        elif board.empty:
            p1score, p2score = score.get_score_pair()
            if p1score == p2score:
                raise Exception("Internal error: player scores equal when board is empty.")
            if p1score > p2score:
                self.outcome = 1
            else:
                self.outcome = -1
        else:
            self.outcome = 0

        self.hash_val = hash(
            (self.board, self.score, self.next_go, self.outcome)
        )

    def __str__(self):
        # Who's go is next
        s = str(int((3 - self.next_go)/2)) + "/"

        # Score
        for x in self.score.score:
            s += ("-" if x < 0 else "") + chr(ord('a') + abs(x))
        
        s += "/"
        
        # Board
        rows = []
        for row in self.board.board:
            r = ""
            blank_counter = 0

            for tile in row:
                if tile == -1:
                    blank_counter += 1
                else:
                    if blank_counter > 0:
                        r += str(blank_counter)
                        blank_counter = 0
                    r += chr(ord('a') + tile)

            if blank_counter > 0:
                r += str(blank_counter)
                blank_counter = 0
            
            rows.append(r)
        s += ".".join(rows)

        return s
            

    def __hash__(self):
        return self.hash_val
    
    def __eq__(self, other):
        return other and self.board == other.board and self.score == other.score and self.next_go == other.next_go and self.outcome == other.outcome
    
    def __ne__(self, other):
        return not self.__eq__(other)

if __name__ == "__main__":

    game = Game(5)

    boards = set()

    boards.add(game.state)
    game.make_move(frozenset(frozenset(0,0)))
    game.make_move(frozenset((4,0)))
    boards.add(game.state)
    game.undo_move()
    game.undo_move()
    boards.add(game.state)
    game.make_move(frozenset((4,0)))
    game.make_move(frozenset((0,0)))
    boards.add(game.state)

    print(boards)

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