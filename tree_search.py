
class TreeSearch:

    def __init__(self, game):
        self.game = game
        self.progress_layers = []
    
    def playout(self, choice_fn, pre_fn=None, record_val=None):
        if pre_fn:
            pre_fn()

        if self.game.state.outcome != 0:
            return self.game.state.outcome
        
        choice = choice_fn()
        if isinstance(choice, tuple):
            chosen_move, _ = choice
        else:
            chosen_move = choice
            
        self.game.make_move(chosen_move)
        val = self.playout(choice_fn, pre_fn, record_val)
        self.game.undo_move()

        if record_val:
            record_val(val)

        return val

    def best_move(self, get_score):
        if self.game.state.outcome != 0:
            return None

        best_score = -float('inf') * self.game.state.next_go
        best_move = None
        
        moves = self.game.get_moves()
        self.progress_layers.append((0, len(moves)))

        for n, move in enumerate(moves):
            self.game.make_move(move)
            score = get_score()
            self.progress_layers[-1] = (n+1, len(moves))
            
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
    
    def get_progress(self):
        if len(self.progress_layers) == 0:
            return 1
        
        progress = 0
        scale = 1
        for p, n in self.progress_layers:
            progress += scale*p/n
            scale *= 1/n
        
        return progress