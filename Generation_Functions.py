import random
from itertools import product
import numpy as np
from TicTacToe import TicTacToe
from ConnectFour import ConnectFour
from Othello import Othello
from Ataxx import Ataxx
from Checkers import Checkers
from State import UNPLAYABLE

NUM_HEURISTICS = 5 # control, mobility, stability, connectivity, tension


#generate sample games by playing random legal moves and evaluate the game state
def eval_sample_positions(game, n_samples=1000, max_steps_per_game=2000):
    evals = []
    choice = random.choice
    control = game.control
    mobility = game.mobility
    stability = game.stability
    connectivity = game.connectivity
    tension = game.tension

    for _ in range(n_samples):
        state = game.initial_state()
        steps_taken = 0
        while not game.game_over(state):
            if max_steps_per_game is not None and steps_taken >= int(max_steps_per_game):
                break

            moves = game.legal_moves(state)
            if not moves:
                break

            move = choice(moves)

            state = game.make_move(state, move)
            steps_taken += 1
            eval = [
                control(state),
                mobility(state),
                stability(state),
                connectivity(state),
                tension(state),
            ]
            evals.append(eval)
            
    return evals

def calculate_label(evals):
    evals_array = np.asarray(evals, dtype=np.float64)
    if evals_array.size == 0:
        return [(0.0, 0.0)] * NUM_HEURISTICS

    means = evals_array.mean(axis=0)
    stand_devs = evals_array.std(axis=0)
    return list(zip(means.tolist(), stand_devs.tolist()))


def calculate_label_online(game, n_samples=250, max_steps_per_game=200):
    choice = random.choice
    control = game.control
    mobility = game.mobility
    stability = game.stability
    connectivity = game.connectivity
    tension = game.tension

    count = 0
    means = np.zeros(NUM_HEURISTICS, dtype=np.float64)
    m2 = np.zeros(NUM_HEURISTICS, dtype=np.float64)

    for _ in range(n_samples):
        state = game.initial_state()
        steps_taken = 0
        while not game.game_over(state):
            if max_steps_per_game is not None and steps_taken >= int(max_steps_per_game):
                break

            moves = game.legal_moves(state)
            if not moves:
                break

            move = choice(moves)
            state = game.make_move(state, move)
            steps_taken += 1

            values = np.array(
                [
                    control(state),
                    mobility(state),
                    stability(state),
                    connectivity(state),
                    tension(state),
                ],
                dtype=np.float64,
            )

            count += 1
            delta = values - means
            means += delta / count
            m2 += delta * (values - means)

    if count == 0:
        return [(0.0, 0.0)] * NUM_HEURISTICS

    stand_devs = np.sqrt(m2 / count)
    return list(zip(means.tolist(), stand_devs.tolist()))


def _iter_variant_combinations(piece_limits, board_sizes, unplayable_proportions, num_turn_values, repeats):
    piece_limits = _normalize_piece_limits(piece_limits)
    board_sizes = _normalize_board_sizes(board_sizes)
    unplayable_proportions = _normalize_unplayable_proportions(unplayable_proportions)
    num_turn_values = _normalize_num_turn_values(num_turn_values)

    for _ in range(num_repeats(repeats)):
        for piece_limit, (rows, cols), (edge_ratio, inner_ratio), num_turns in product(piece_limits, board_sizes, unplayable_proportions, num_turn_values):
            yield piece_limit, rows, cols, edge_ratio, inner_ratio, num_turns


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


def _normalize_piece_limits(piece_limits):
    normalized = []
    for limit in piece_limits:
        if limit is None:
            normalized.append(None)
            continue

        normalized_limit = int(limit)
        if normalized_limit < 1:
            raise ValueError("piece limits must be positive integers or None")
        normalized.append(normalized_limit)

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


def _normalize_num_turn_values(num_turn_values):
    normalized = []
    for value in num_turn_values:
        normalized_value = int(value)
        if normalized_value == 0:
            raise ValueError("num_turn values must be non-zero")
        normalized.append(normalized_value)

    return normalized

def generate_tic_tac_toe(
    num_variants=3,
    piece_limits=None,
    board_sizes=None,
    unplayable_proportions=None,
    num_turn_values=None,
    n_samples=250, # number of random games to simulate for evaluation per variant
    max_steps_per_game=200, # maximum number of moves to simulate in each random game to prevent infinite loops in complex variants
):
    # With this configuration, there should be 81 total variants (3 piece limits x 3 board sizes x 3 unplayable proportions x 3 num_turn_values), num_variants amount of these are then generated
    if piece_limits is None:
        piece_limits = [None, 4, 6]
    if board_sizes is None:
        board_sizes = [(3, 3), (5, 5), (6, 8)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.1, 0.1), (0.15, 0.15)] # (edge_unplayable_ratio, inner_unplayable_ratio)
    if num_turn_values is None:
        num_turn_values = [1, 2, -2, 3, -3]

    X = []
    Y = []
    # counter = 0 # DEBUGGING
    for piece_limit, rows, cols, edge_ratio, inner_ratio, num_turns in _iter_variant_combinations(
        piece_limits, board_sizes, unplayable_proportions, num_turn_values, num_variants
    ):
        pl = 9999999 if piece_limit is None else piece_limit
        max_in_a_row = min(rows, cols, pl)
        sampled_in_a_row = random.randint(3, max_in_a_row)
        in_a_row = max_in_a_row if random.random() < 0.5 else sampled_in_a_row

        game = TicTacToe(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            in_a_row=in_a_row,
            max_pieces_per_player=piece_limit,
            num_turns=num_turns,
        )
        initial_state = game.initial_state()
        playable_tiles = int(np.sum(initial_state.board != UNPLAYABLE))
        piece_limit_value = 0 if piece_limit is None else piece_limit

        #    rows,      cols,      num_pieces,     max_pieces_per_player, num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win, turns_per_block
        x = [game.rows, game.cols, playable_tiles, piece_limit_value,     1,                 1,              0,        0,          edge_ratio,      inner_ratio,      in_a_row,        num_turns]
        y = calculate_label_online(game, n_samples=n_samples, max_steps_per_game=max_steps_per_game)
        X.append(x)
        Y.append(y)
        # counter += 1 # DEBUGGING
        # print(f"Completed {counter} variants") # DEBUGGING
        print(f"Generated TicTacToe variant with rows={rows}, cols={cols}, piece_limit={piece_limit}, edge_unplayable_ratio={edge_ratio}, inner_unplayable_ratio={inner_ratio}, in_a_row={in_a_row}, num_turns={num_turns}")
    
    return X, Y

def generate_connect_four(
    num_variants=10,
    piece_limits=None,
    board_sizes=None,
    unplayable_proportions=None,
    num_turn_values=None,
    n_samples=250,
    max_steps_per_game=200,
):
    if piece_limits is None:
        piece_limits = [None, 8, 14]
    if board_sizes is None:
        board_sizes = [(6, 7), (7, 9), (11, 11)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.1, 0.1), (0.15, 0.15)]
    if num_turn_values is None:
        num_turn_values = [1, 2, -2, 3, -3]

    X = []
    Y = []
    for piece_limit, rows, cols, edge_ratio, inner_ratio, num_turns in _iter_variant_combinations(
        piece_limits, board_sizes, unplayable_proportions, num_turn_values, num_variants
    ):
        pl = 9999999 if piece_limit is None else piece_limit
        max_in_a_row = min(rows, cols, pl)
        sampled_in_a_row = random.randint(3, max_in_a_row)
        in_a_row = 4 if random.random() < 0.5 else sampled_in_a_row

        game = ConnectFour(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            in_a_row=in_a_row,
            max_pieces_per_player=piece_limit,
            num_turns=num_turns,
        )
        initial_state = game.initial_state()
        playable_tiles = int(np.sum(initial_state.board != UNPLAYABLE))
        piece_limit_value = 0 if piece_limit is None else piece_limit

        #    rows,      cols,      num_pieces,     max_pieces_per_player, num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win, turns_per_block
        x = [game.rows, game.cols, playable_tiles, piece_limit_value,     1,                 1,              0,        0,          edge_ratio,      inner_ratio,      in_a_row,        num_turns]
        y = calculate_label_online(game, n_samples=n_samples, max_steps_per_game=max_steps_per_game)
        X.append(x)
        Y.append(y)
        print(f"Generated ConnectFour variant with rows={rows}, cols={cols}, piece_limit={piece_limit}, edge_unplayable_ratio={edge_ratio}, inner_unplayable_ratio={inner_ratio}, in_a_row={in_a_row}, num_turns={num_turns}")
    
    return X, Y

def generate_othello(
    num_variants=10,
    piece_limits=None,
    board_sizes=None,
    unplayable_proportions=None,
    num_turn_values=None,
    n_samples=250,
    max_steps_per_game=200,
):
    if piece_limits is None:
        piece_limits = [None, 8, 12]
    if board_sizes is None:
        board_sizes = [(8, 8), (10, 10), (12, 12)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.1, 0.1), (0.15, 0.15)]
    if num_turn_values is None:
        num_turn_values = [1, 2, -2, 3, -3]

    X = []
    Y = []
    for piece_limit, rows, cols, edge_ratio, inner_ratio, num_turns in _iter_variant_combinations(
        piece_limits, board_sizes, unplayable_proportions, num_turn_values, num_variants
    ):
        game = Othello(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            max_pieces_per_player=piece_limit,
            num_turns=num_turns,
        )
        initial_state = game.initial_state()
        playable_tiles = int(np.sum(initial_state.board != UNPLAYABLE))
        piece_limit_value = 0 if piece_limit is None else piece_limit

        #    rows,      cols,      num_pieces,     max_pieces_per_player, num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win, turns_per_block
        x = [game.rows, game.cols, playable_tiles, piece_limit_value,     1,                 1,              1,        1,          edge_ratio,      inner_ratio,      0,               num_turns]
        y = calculate_label_online(game, n_samples=n_samples, max_steps_per_game=max_steps_per_game)
        X.append(x)
        Y.append(y)
        print(f"Generated Othello variant with rows={rows}, cols={cols}, piece_limit={piece_limit}, edge_unplayable_ratio={edge_ratio}, inner_unplayable_ratio={inner_ratio}, num_turns={num_turns}")
    return X, Y

def generate_ataxx(
    num_variants=10,
    piece_limits=None,
    board_sizes=None,
    unplayable_proportions=None,
    num_turn_values=None,
    n_samples=250,
    max_steps_per_game=200,
):
    print("Generating Ataxx variants...") # DEBUGGING
    if piece_limits is None:
        piece_limits = [None, 45, 75]
    if board_sizes is None:
        board_sizes = [(7, 7), (8, 8), (11, 11)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.1, 0.1), (0.15, 0.15)]
    if num_turn_values is None:
        num_turn_values = [1, 2, -2, 3, -3]

    X = []
    Y = []
    for piece_limit, rows, cols, edge_ratio, inner_ratio, num_turns in _iter_variant_combinations(
        piece_limits, board_sizes, unplayable_proportions, num_turn_values, num_variants
    ):
        game = Ataxx(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            max_pieces_per_player=piece_limit,
            num_turns=num_turns,
        )
        initial_state = game.initial_state()
        playable_tiles = int(np.sum(initial_state.board != UNPLAYABLE))
        piece_limit_value = 0 if piece_limit is None else piece_limit

        #    rows,      cols,      num_pieces,     max_pieces_per_player, num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win, turns_per_block
        x = [game.rows, game.cols, playable_tiles, piece_limit_value,     1,                 0,              1,        1,          edge_ratio,      inner_ratio,      0,               num_turns]
        y = calculate_label_online(game, n_samples=n_samples, max_steps_per_game=max_steps_per_game)
        X.append(x)
        Y.append(y)

        print(f"Generated Ataxx variant with rows={rows}, cols={cols}, piece_limit={piece_limit}, edge_unplayable_ratio={edge_ratio}, inner_unplayable_ratio={inner_ratio}, num_turns={num_turns}")
    
    return X, Y


#To generate "lobsided" checkers set keep_pieces to True when changing boardsize
#To generate bigger checkers, set keep pieces to False when changing boardsize
def generate_checkers(
    num_variants=10,
    piece_limits=None,
    keep_pieces=True,
    board_sizes=None,
    unplayable_proportions=None,
    num_turn_values=None,
    n_samples=250,
    max_steps_per_game=200,
):
    if piece_limits is None:
        piece_limits = [None, 7, 10]
    if board_sizes is None:
        board_sizes = [(8, 8), (10, 10), (13, 13)]
    if unplayable_proportions is None:
        unplayable_proportions = [(0.0, 0.0), (0.1, 0.1), (0.15, 0.15)]
    if num_turn_values is None:
        num_turn_values = [1, 2, -2, 3, -3]

    X = []
    Y = []
    for piece_limit, rows, cols, edge_ratio, inner_ratio, num_turns in _iter_variant_combinations(
        piece_limits, board_sizes, unplayable_proportions, num_turn_values, num_variants
    ):
        game = Checkers(
            rows=rows,
            cols=cols,
            keep_pieces=keep_pieces,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            max_pieces_per_player=piece_limit,
            num_turns=num_turns,
        )
        initial_state = game.initial_state()
        num_starting_pieces = int(np.sum((initial_state.board != 0) & (initial_state.board != UNPLAYABLE)))
        piece_limit_value = 0 if piece_limit is None else piece_limit

        #    rows,      cols,      num_pieces,          max_pieces_per_player, num_unique_pieces, placement_game, captures, space_game, edge_unplayable, inner_unplayable, in_a_row_to_win, turns_per_block
        x = [game.rows, game.cols, num_starting_pieces, piece_limit_value,     2,                 0,              1,        0,          edge_ratio,      inner_ratio,      0,               num_turns]
        y = calculate_label_online(game, n_samples=n_samples, max_steps_per_game=max_steps_per_game)
        X.append(x)
        Y.append(y)
        print(f"Generated Checkers variant with rows={rows}, cols={cols}, piece_limit={piece_limit}, edge_unplayable_ratio={edge_ratio}, inner_unplayable_ratio={inner_ratio}, num_turns={num_turns}")
    return X,Y

