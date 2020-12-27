import dgl
import numpy as np
import time
import torch
from queue import PriorityQueue
from random import shuffle
from torch import tensor
from tqdm import tqdm
from typing import NamedTuple

class Pos(NamedTuple):
    row: int
    col: int
    
    def left(self):
        return Pos(self.row-1, self.col)
    def right(self):
        return Pos(self.row+1, self.col)
    def up(self):
        return Pos(self.row, self.col-1)
    def down(self):
        return Pos(self.row, self.col+1)
    
    def __str__(self):
        return "{0},{1}".format(self.row, self.col)
    
    def __repr__(self):
        return self.__str__()

class Game:
    """
    Defines the game logic, state and string representation.
    """

    def __init__(self, base=7, state=None):
        if base % 2 == 0 or base < 3:
            raise Exception("Invalid base number. Must be odd and > 2.")

        self.base = base

        if state == None:
            self.state = State(
                board=Board(base),
                score=Score(base),
                next_go=1 # Player 1 or -1 to play?
            )
        else:
            self.base = state.score.base
            self.state = state
        
        self.search_game = None # Game object used by agents to search the state tree

    def get_moves(self):
        """
        Returns all possible next moves for a game state.
        """
        return self.state.board.get_moves()
    
    def get_takable(self):
        """
        Returns a set of (row, column) positions of tiles which are corner tiles i.e. can be taken next move.
        """
        return self.state.board.get_takable()
    
    def get_at(self, a, b=None):
        """
        Returns the value of a tile at the position specified or -1 if no tile exists there.
        """
        return self.state.board.get_at(a, b)
    
    def get_score(self, i):
        """
        Returns the current score of a colour.
        """
        return self.state.score.get_score(i)
    
    def over(self):
        return self.state.outcome != 0

    def make_move(self, move):
        """
        Creates a new State (the result after the move has been made).
        The states are memoised so they don't need to be recalculated.
        """
        if move.next_state:
            self.state = move.next_state
            return

        color = self.get_at(next(iter(move)))
        board = self.state.board.make_move(move)
        score = self.state.score.make_move(move, color, self.state.next_go)
        
        self.state = State(
            board=board,
            score=score,
            next_go=-self.state.next_go,
            parent=self.state,
            move=move
        )

        move.next_state = self.state
    
    def undo_move(self):
        """
        Returns the game state to the previous state.
        """
        self.state = self.state.parent
    
    def reset_search_game(self):
        """
        Ensures the state of self.search_game is the same as the current state.
        """
        if not self.search_game:
            self.search_game = Game(self.base, self.state)
        self.search_game.state = self.state
    
    @staticmethod
    def from_str(s):
        """
        Creates a game beginning from the string representation of a state.
        """

        state = State.from_str(s)

        return Game(
            state.score.base,
            state
        )

class Move:

    def __init__(self, move_set):
        self.tiles = tuple(map(lambda p: Pos(*p), sorted(list(move_set))))
        self.next_state = None

    def __getitem__(self, index):
        return self.tiles[index]
    
    def __len__(self):
        return len(self.tiles)

    def __hash__(self):
        return self.tiles.__hash__()
    
    def __eq__(self, other):
        return other and self.tiles == other.tiles
    
    def __str__(self):
        strs = ["{},{}".format(tile.row, tile.col) for tile in self.tiles]
        return ";".join(strs)
    
    def __repr__(self):
        return self.__str__()

class Board:
    """
    Immutable representation of the game board and all the tiles.
    """
    def __init__(self, base, board=None):
        self.base = base

        if board == None:
            # Randomly generate a board assignment
            tiles = [i % base for i in range(base**2)]
            shuffle(tiles)
            board = [[tiles.pop() for _ in range(base)] for _ in range(base)]
        
        self.board = tuple([tuple(row) for row in board])
        
        # Empty if all cells have value -1
        self.empty = all(
            board[row][col] == -1
            for row in range(base)
            for col in range(base)
        )
        
        # List of all corner tiles
        self.takable = [
            Pos(row, col)
            for row in range(self.base)
            for col in range(self.base)
            if self.get_at(row, col) >= 0
            and (self.get_at(row-1, col) == -1 or self.get_at(row+1, col) == -1)
            and (self.get_at(row, col-1) == -1 or self.get_at(row, col+1) == -1)
        ]

        # List of takable tiles for each color.
        takable_colors = [
            [
                tile
                for tile in self.get_takable()
                if self.get_at(tile) == i
            ]
            for i in range(self.base)
        ]

        # All possible next moves.
        self.moves = [
            comb
            for tiles in takable_colors
            for comb in self._all_combs(tiles)
        ]

        self.hash_val = hash(self.board)
    
    @staticmethod
    def _all_combs(a):
        """
        Returns a list of all ways to take any number > 0 of members of a.
        """
        combs = []
        for b in range(1, 2**len(a)):
            combs.append(Move({
                a[i]
                for i in range(len(a))
                if (b >> i) & 1 == 1
            }))
        
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
        if base%2 != 1 or base < 3 or base > 25:
            raise Exception("Base number must be odd and between 2 and 26.")

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
    
    def __str__(self):
        rows = []
        for row in self.board:
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
        return ".".join(rows)

    def __hash__(self):
        return self.hash_val
    
    def __eq__(self, other):
        return other and self.board == other.board
    
    def __ne__(self, other):
        return not self.__eq__(other)

class Score:
    """
    Immutable representation of the score of the game.
    """
    def __init__(self, base, score=None):
        self.base = base

        if score == None:
            score = tuple([0 for _ in range(base)])
        else:
            score = tuple(score)
        
        self.score = score
        
        self.validate(score)
        if len(score) != base:
            raise Exception("The score must have the same number of items as the base number.\nBase: {0}\nItems: {1}".format(base, len(score)))

        # 1 if player1 has captured all of a colour, -1 if player2, else 0
        self.player_with_all = sum([int(x/self.base) for x in self.score])

        # The number of colors the two players have the majority in.
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
    
    def get_score(self, i):
        return self.score[i]
    
    def get_score_pair(self):
        return self.score_pair
    
    @staticmethod
    def validate(score):
        base = len(score)
        if base%2 != 1 or base < 3 or base > 25:
            raise Exception("Base number must be odd and between 2 and 26.")

        if sum([abs(x) > base for x in score]) > 0:
            raise Exception("Score cannot have values > base number.")
        
        if sum([abs(x) == base for x in score]) > 1:
            raise Exception("At most one score value can be the base number.")

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
    
    def __str__(self):
        s = ""
        for x in self.score:
            s += ("-" if x < 0 else "") + chr(ord('a') + abs(x))

        return s

    def __hash__(self):
        return self.hash_val
    
    def __eq__(self, other):
        return other and self.score == other.score
    
    def __ne__(self, other):
        return not self.__eq__(other)

class Relation:
    HIDDEN_BY = 0
    REVEALS = 1
    DIAGONAL = 2
    TAKABLE = 3
    SAME_COLOR = 4

class State:
    """
    Immutable representation of the game state.
    """

    def __init__(self, board, score, next_go, parent=None, move=None):
        self.board = board
        self.score = score
        self.next_go = next_go
        self.parent = parent
        self.move = move
        self.dgl_graph = None

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
        s += str(self.score) + "/"
        
        # Board
        s += str(self.board)

        return s
    
    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.hash_val
    
    def __eq__(self, other):
        return other and \
            isinstance(other, State) and \
                self.board == other.board and \
                    self.score == other.score and \
                        self.next_go == other.next_go and \
                            self.outcome == other.outcome
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def get_game_str(self):
        if self.parent == None:
            return str(self) + "|"

        prev = self.parent.get_game_str()
        if prev[-1] != "|":
            prev += "/"

        return prev + str(self.move)

    @staticmethod
    def from_str(s):
        """
        Parses the game state from a string representation.
        Example starting state for a 3x3 board:
            1/aaa/aab.cba.cbc
            ^ ^   ^
            | |   board state with rows separated by . and a=0, b=1,... for the tile values
            | score state with a=0, b=1,... and the index corresponding to color.
            the player who's turn it is next; can be 1 or 2.
        A negative score value is preceeded by -
        An empty cell or a sequence of empty cells on the same row is denoted by the number of consequtive empty cells.
        """
        parts = s.split('/')
        if len(parts) != 3:
            raise Exception("Must have 3 parts separated by '/'.")

        if parts[0] not in ["1", "2"]:
            raise Exception("Expected first part to be 1 or 2; got {}.".format(parts[0]))

        next_go = 3 - 2*int(parts[0])
        score = Score.from_str(parts[1])
        board = Board.from_str(parts[2])

        if score.base != board.base:
            raise Exception("Score and board must have the same base number.")

        return State(board, score, next_go)

    def get_feature_vector(self, pos):
        tile = self.board.get_at(pos)
        score_pair = self.score.get_score_pair()

        return [
            self.board.base - self.score.get_score(tile)*self.next_go,
            self.board.base + self.score.get_score(tile)*self.next_go,
            (score_pair[0] - score_pair[1])*self.next_go
        ]
    
    def to_dgl_graph(self):
        if self.dgl_graph != None:
            return self.dgl_graph

        size_bound = self.board.base**3 + 16*self.board.base**2
       
        class Edges:
            def __init__(self):
                self.etypes = [0]*size_bound
                self.src = [0]*size_bound
                self.dst = [0]*size_bound
                self.size = 0
            
            def add_edge(self, u, v, rel_id, both_ways=False):
                if self.size == size_bound:
                    print(size_bound)
                self.etypes[self.size] = rel_id
                self.src[self.size] = u
                self.dst[self.size] = v
                self.size += 1
                if both_ways:
                    if self.size == size_bound:
                        print(size_bound)
                    self.etypes[self.size] = rel_id
                    self.src[self.size] = v
                    self.dst[self.size] = u
                    self.size += 1

            def get_etypes(self):
                return self.etypes[:self.size]
            
            def get_src_dst(self):
                return (
                    self.src[:self.size],
                    self.dst[:self.size]
                )
            
            # def gpu_tensors(self):
            #     # t = get_time()
            #     src_tensor.copy_(src_cpu)
            #     dst_tensor.copy_(dst_cpu)
            #     # print("copy time:", diff_str(t))

            #     return (
            #         src_tensor[:self.size],
            #         dst_tensor[:self.size]
            #     )
            
            # def cpu_tensors(self):
            #     return (
            #         src_cpu[:self.size],
            #         dst_cpu[:self.size]
            #     )

        edges = Edges()

        features = []

        index_map = {} # from board pos to node index
        pos_order = [] # all tile positions in order of node index

        takable_set = set()
        color_sets = {}

        # Assign the index map and build the node features
        # [player’s colour score, opponent’s colour score, score difference]
        for row in range(self.board.base):
            for col in range(self.board.base):
                pos = Pos(row, col)
                tile = self.board.get_at(pos)
                if tile > -1:
                    index_map[pos] = len(features)
                    pos_order.append(pos)

                    color_set = color_sets.get(tile, set())
                    color_set.add(pos)
                    color_sets[tile] = color_set

                    if pos in self.board.get_takable():
                        takable_set.add(pos)

                    features.append(self.get_feature_vector(pos))

        # Add all positional relations
        for pos, index in index_map.items():
            if pos not in self.board.get_takable():
                
                blocked_h = self.board.get_at(pos.left()) > -1 and self.board.get_at(pos.right()) > -1
                blocked_v = self.board.get_at(pos.up()) > -1 and self.board.get_at(pos.down()) > -1
                
                assert blocked_h or blocked_v

                if blocked_h:
                    left = index_map[pos.left()]
                    right = index_map[pos.right()]

                    edges.add_edge(index, left, Relation.HIDDEN_BY)
                    edges.add_edge(index, right, Relation.HIDDEN_BY)
                    edges.add_edge(left, index, Relation.REVEALS)
                    edges.add_edge(right, index, Relation.REVEALS)
                
                if blocked_v:
                    up = index_map[pos.up()]
                    down = index_map[pos.down()]

                    edges.add_edge(index, up, Relation.HIDDEN_BY)
                    edges.add_edge(index, down, Relation.HIDDEN_BY)
                    edges.add_edge(up, index, Relation.REVEALS)
                    edges.add_edge(down, index, Relation.REVEALS)
                
                if blocked_h and blocked_v:
                    edges.add_edge(left, up, Relation.DIAGONAL, both_ways=True)
                    edges.add_edge(up, right, Relation.DIAGONAL, both_ways=True)
                    edges.add_edge(right, down, Relation.DIAGONAL, both_ways=True)
                    edges.add_edge(down, left, Relation.DIAGONAL, both_ways=True)

        # Densely connect all same color nodes and all takable nodes.
        for relation_set, rel_id in [(takable_set, Relation.TAKABLE)] + [(color_set, Relation.SAME_COLOR) for color_set in color_sets.values()]:
            for u in relation_set:
                for v in relation_set:
                    if u != v:
                        edges.add_edge(index_map[u], index_map[v], rel_id)

        graph = dgl.graph(edges.get_src_dst())

        if graph.num_nodes() == 0: # case that no edges are added
            graph.add_nodes(len(features))

        graph.edata.update({"rel_type": tensor(edges.get_etypes())})
        graph.ndata.update({"features": tensor(features, dtype=torch.float)})
        graph.ndata.update({"position": tensor(pos_order)})

        self.dgl_graph = graph

        return graph

from agents.random_agent import RandomAgent
if __name__ == "__main__":

    st = get_time()

    times = []
    for i in range(1000):
        game = Game.from_str("1/aaaaaaa/fdcfaaa.fafgbde.eedggec.accbbfb.fegdfba.gdeccbc.ddabegg")
        agent = RandomAgent(game)
        while not game.over():
            t = get_time()
            game.make_move(agent.select_move())
            times.append(get_time() - t)
    
    print(pprint("total:", diff_str(st)))
    
    print(len(times))
    print(sum(times))
    print(1000*sum(times)/len(times))