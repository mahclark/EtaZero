class TreeSearch:
    """
    A class for functions useful for tree search.
    One should be instantiated per agent so individual progress can be measured.
    """

    def __init__(self, game):
        self.game = game  # Must be the same game as the one used by choice_fn and get_score below

        # Pairs of (p, n) where n is the number of tasks and p (<= n) is the number of tasks completed.
        # If a pair has a preceeding pair, its progress corresponds to one task of the preceeding pair.
        # E.g. [(2,4), (1,2)] corresponds to 5/8 total progress
        self.progress_layers = []

    def playout(self, choice_fn, pre_fn=None, terminate_fn=None, record_val=None):
        """
        Plays a game to completion, given a function to choose moves.
        Returns the result and undos all moves made.
        """

        # Perform any setup for each recursion
        if pre_fn:
            pre_fn()

        # Check if the game has finished
        if self.game.state.outcome != 0:
            return self.game.state.outcome

        # Check if we need to terminate
        if terminate_fn and terminate_fn():
            return None

        # Chose, make the move and recurse
        chosen_move = choice_fn()

        self.game.make_move(chosen_move)
        val = self.playout(choice_fn, pre_fn, terminate_fn, record_val)
        self.game.undo_move()

        # Record the value if necessary
        if record_val:
            record_val(val)

        return val

    def best_move_and_score(self, get_score):
        """
        Calculates the score for all moves and returns the best move and its corresponding score given a score function.
        Assumes that player1 prefers a positive score and player2 prefers a negative score.
        Adds and completes a new task to progress_layers.
        """

        if self.game.state.outcome != 0:
            return (None, None)

        best_score = -float("inf") * self.game.state.next_go
        best_move = None

        moves = self.game.get_moves()
        self.progress_layers.append((0, len(moves)))

        for n, move in enumerate(moves):
            self.game.make_move(move)
            score = get_score()
            self.progress_layers[-1] = (n + 1, len(moves))

            if self.game.state.next_go == -1:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

            self.game.undo_move()

        del self.progress_layers[-1]

        return best_move, best_score

    def best_move(self, get_score):
        """
        Required so the function can be used as choice_fn in self.playout().
        """
        return self.best_move_and_score(get_score)[0]

    def get_progress(self, progress_layers=None):
        """
        Returns the total progress as defined in constructor above.
        """
        if not progress_layers:
            progress_layers = self.progress_layers

        if len(progress_layers) == 0:
            return 1

        progress = 0
        scale = 1
        for p, n in progress_layers:
            progress += scale * p / n
            scale *= 1 / n

        return progress
