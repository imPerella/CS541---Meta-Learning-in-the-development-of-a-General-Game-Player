import random
from TicTacToe import TicTacToe
from ConnectFour import ConnectFour
from Othello import Othello
from Ataxx import Ataxx
from Checkers import Checkers
from State import *

#generate sample games by playing random legal moves and evaluate the game state
def eval_sample_positions(game, n_samples=1000):
    evals = []

    for i in range(n_samples):
        state = game.initial_state()
        while not game.game_over(state):
            moves = game.legal_moves(state)

            if moves != None:
                move = random.choice(moves)
            else:
                move = None

            state = game.make_move(state, move)
            eval = [game.control(state), game.mobility(state), game.stability(state), game.connectivity(state), game.tension(state)]
            evals.append(eval)
            
    return evals

def calculate_label (evals):
    means = np.sum(evals, axis=0)/len(evals)
    squared_diffs = []
    for e in evals:
        squared_diff = (np.array(e) - np.array(means))**2
        squared_diffs.append(squared_diff)

    inroot = np.sum(squared_diffs, axis=0)/len(evals)
    stand_devs = np.sqrt(inroot)

    label = list(zip(means, stand_devs))
    
    return label

def generate_tic_tac_toe(num_variants=10):
    X = []
    Y = []
    for i in range(3, 3+num_variants):
        game = TicTacToe(rows=i, cols=i)
        #[rows, cols, num_pieces, num_unique_pieces, gravity, placement_game, captures,  forced_captures, inarow_game, space_game, capture_game]
        x = [game.rows, game.cols, 9, 1, 0, 1, 0, 0, 1, 0, 0]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X, Y

def generate_connect_four(num_variants=10):
    X = []
    Y = []
    for i in range(7, 7+num_variants):
        game = ConnectFour(rows=i-1, cols=i)
        #[rows, cols, num_pieces, num_unique_pieces, gravity, placement_game, captures,  forced_captures, inarow_game, space_game, capture_game]
        x = [game.rows, game.cols, 1, 1, 1, 0, 0, 1, 0, 0]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X, Y

def generate_othello(num_variants=10):
    X = []
    Y = []
    for i in range(8, 8+num_variants):
        game = Othello(rows=i, cols=i)
        #[rows, cols, num_pieces, num_unique_pieces, gravity, placement_game, captures,  forced_captures, inarow_game, space_game, capture_game]
        x = [game.rows, game.cols, 1, 0, 1, 1, 1, 0, 1, 0]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X, Y

def generate_ataxx(num_variants=10):
    X = []
    Y = []
    for i in range(7, 7+num_variants):
        game = Ataxx(rows=i, cols=i)
        #[rows, cols, num_pieces, num_unique_pieces, gravity, placement_game, captures,  forced_captures, inarow_game, space_game, capture_game]
        x = [game.rows, game.cols, 1, 0, 1, 1, 0, 0, 1, 0]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X, Y


#To generate "lobsided" checkers set keep_pieces to True when changing boardsize
#To generate bigger checkers, set keep pieces to False when changing boardsize
def generate_checkers(num_variants=10, keep_pieces=True):
    X = []
    Y = []
    for i in range(8, 8+num_variants):
        game = Checkers(rows=i, cols=i, keep_pieces=keep_pieces)
        #[rows, cols, num_pieces, num_unique_pieces, gravity, placement_game, captures,  forced_captures, inarow_game, space_game, capture_game]
        x = [game.rows, game.cols, 2, 0, 0, 1, 1, 0, 0, 1]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X,Y

