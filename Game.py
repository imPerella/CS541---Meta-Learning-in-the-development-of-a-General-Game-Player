
class Game:

    # I implemented the 3 functions below here as they are the same for all games
    def _validate_num_turns(self, num_turns):
        normalized = int(num_turns)
        if normalized == 0:
            raise ValueError("num_turns must be non-zero")
        return normalized

    def _turn_quota(self, player):
        if self.num_turns > 0:
            return self.num_turns if player == 1 else 1
        return abs(self.num_turns) if player == -1 else 1

    def _next_turn(self, current_player, turns_remaining):
        if turns_remaining > 1:
            return current_player, turns_remaining - 1

        next_player = -current_player
        return next_player, self._turn_quota(next_player)

    def initial_state(self):
        raise NotImplementedError
    
    def legal_moves(self, state):
        raise NotImplementedError
    
    def make_move(self, state, move):
        raise NotImplementedError
    
    def game_over(self, state):
        raise NotImplementedError
    
    #heuristic calculation
    def control(self, state):
        raise NotImplementedError
    
    def mobility(self, state):
        raise NotImplementedError
    
    def stability (self, state):
        raise NotImplementedError
    
    def connectivity(self, state):
        raise NotImplementedError
    
    def tension(self, state):
        raise NotImplementedError
    

