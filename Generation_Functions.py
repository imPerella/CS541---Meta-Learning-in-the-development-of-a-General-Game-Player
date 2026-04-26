import random
from itertools import product
import numpy as np
from TicTacToe import TicTacToe
from ConnectFour import ConnectFour
from Othello import Othello
from Ataxx import Ataxx
from Checkers import Checkers
from State import UNPLAYABLE

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


def _iter_variant_combinations(board_sizes, unplayable_proportions, repeats):
    board_sizes = _normalize_board_sizes(board_sizes)
    unplayable_proportions = _normalize_unplayable_proportions(unplayable_proportions)

    for _ in range(num_repeats(repeats)):
        for (rows, cols), (edge_ratio, inner_ratio) in product(board_sizes, unplayable_proportions):
            yield rows, cols, edge_ratio, inner_ratio


def num_repeats(repeats):
    return max(0, int(repeats))


def _normalize_board_sizes(board_sizes):
    normalized = []
    for size in board_sizes:
        if isinstance(size, int):
            normalized.append((size, size))
        else:
            rows, cols = size
            normalized.append((int(rows), int(cols)))

    return normalized


def _normalize_unplayable_proportions(unplayable_proportions):
    normalized = []
    for value in unplayable_proportions:
        if isinstance(value, (int, float)):
            normalized.append((float(value), float(value)))
        else:
            edge_ratio, inner_ratio = value
            normalized.append((float(edge_ratio), float(inner_ratio)))

    return normalized

def generate_tic_tac_toe(
    num_variants=10,
    board_sizes=None,
    unplayable_proportions=None,
):
    if board_sizes is None:
        board_sizes = [(3, 3), (4, 4), (5, 5)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.15)] # (edge_unplayable_ratio, inner_unplayable_ratio)

    X = []
    Y = []
    for rows, cols, edge_ratio, inner_ratio in _iter_variant_combinations(
        board_sizes, unplayable_proportions, num_variants
    ):
        max_in_a_row = min(rows, cols)
        sampled_in_a_row = random.randint(3, max_in_a_row)
        in_a_row = max_in_a_row if random.random() < 0.5 else sampled_in_a_row

        game = TicTacToe(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            in_a_row=in_a_row,
        )
        initial_state = game.initial_state()
        playable_tiles = int(np.sum(initial_state.board != UNPLAYABLE))

        #    rows,      cols,      num_pieces,    num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win
        x = [game.rows, game.cols, playable_tiles, 1,                 1,              0,        0,          edge_ratio,       inner_ratio,     in_a_row]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X, Y

def generate_connect_four(
    num_variants=10,
    board_sizes=None,
    unplayable_proportions=None,
):
    if board_sizes is None:
        board_sizes = [(6, 7), (7, 8), (8, 9)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.15)]

    X = []
    Y = []
    for rows, cols, edge_ratio, inner_ratio in _iter_variant_combinations(
        board_sizes, unplayable_proportions, num_variants
    ):
        max_in_a_row = min(rows, cols)
        sampled_in_a_row = random.randint(3, max_in_a_row)
        in_a_row = 4 if random.random() < 0.5 else sampled_in_a_row

        game = ConnectFour(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            in_a_row=in_a_row,
        )
        initial_state = game.initial_state()
        playable_tiles = int(np.sum(initial_state.board != UNPLAYABLE))

        #    rows,      cols,      num_pieces,    num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win
        x = [game.rows, game.cols, playable_tiles, 1,                 1,              0,        0,          edge_ratio,       inner_ratio,     in_a_row]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X, Y

def generate_othello(
    num_variants=10,
    board_sizes=None,
    unplayable_proportions=None,
):
    if board_sizes is None:
        board_sizes = [(8, 8), (10, 10), (12, 12)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.08, 0.08), (0.15, 0.12)]

    X = []
    Y = []
    for rows, cols, edge_ratio, inner_ratio in _iter_variant_combinations(
        board_sizes, unplayable_proportions, num_variants
    ):
        game = Othello(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
        )
        initial_state = game.initial_state()
        playable_tiles = int(np.sum(initial_state.board != UNPLAYABLE))

        #    rows,      cols,      num_pieces,    num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win
        x = [game.rows, game.cols, playable_tiles, 1,                 1,              1,        1,          edge_ratio,       inner_ratio,     0]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X, Y

def generate_ataxx(
    num_variants=10,
    board_sizes=None,
    unplayable_proportions=None,
):
    if board_sizes is None:
        board_sizes = [(7, 7), (8, 8), (9, 9)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.15)]

    X = []
    Y = []
    for rows, cols, edge_ratio, inner_ratio in _iter_variant_combinations(
        board_sizes, unplayable_proportions, num_variants
    ):
        game = Ataxx(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
        )
        initial_state = game.initial_state()
        playable_tiles = int(np.sum(initial_state.board != UNPLAYABLE))

        #    rows,      cols,      num_pieces,    num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win
        x = [game.rows, game.cols, playable_tiles, 1,                 0,              1,        1,          edge_ratio,       inner_ratio,     0]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X, Y


#To generate "lobsided" checkers set keep_pieces to True when changing boardsize
#To generate bigger checkers, set keep pieces to False when changing boardsize
def generate_checkers(
    num_variants=10,
    keep_pieces=True,
    board_sizes=None,
    unplayable_proportions=None,
):
    if board_sizes is None:
        board_sizes = [(8, 8), (10, 10), (12, 12)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.08, 0.08), (0.15, 0.12)]

    X = []
    Y = []
    for rows, cols, edge_ratio, inner_ratio in _iter_variant_combinations(
        board_sizes, unplayable_proportions, num_variants
    ):
        game = Checkers(
            rows=rows,
            cols=cols,
            keep_pieces=keep_pieces,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
        )
        initial_state = game.initial_state()
        num_starting_pieces = int(np.sum((initial_state.board != 0) & (initial_state.board != UNPLAYABLE)))

        #    rows,      cols,      num_pieces,          num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win
        x = [game.rows, game.cols, num_starting_pieces, 2,                 0,              1,        0,          edge_ratio,       inner_ratio,     0]
        y = calculate_label(eval_sample_positions(game))
        X.append(x)
        Y.append(y)
    
    return X,Y

