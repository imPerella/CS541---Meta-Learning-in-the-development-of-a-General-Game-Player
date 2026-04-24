
class Game:

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
    

