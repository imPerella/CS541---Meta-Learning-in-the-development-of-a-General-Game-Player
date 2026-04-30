from pathlib import Path

import numpy as np
import torch

import model # includes HeuristicMetaRegressor and load_selected_datasets
from Generation_Functions import calculate_label_online
from Player import choose_move
from TicTacToe import TicTacToe
from ConnectFour import ConnectFour
from Othello import Othello
from Ataxx import Ataxx
from Checkers import Checkers


NUM_HEURISTICS = 5

# Use "" or None to train a new model using the datasets specified in model.DATASETS, otherwise provide a path to a pre-trained model
MODEL_PATH = model.OUTPUT_DIR / "model_11111.pt"

# len=13, [game_id, rows, cols, num_pieces, max_pieces_per_player, num_unique_pieces, placement_game, captures, space_game, edge_unplayable_ratio, inner_unplayable_ratio, in_a_row_to_win, turns_per_block]
testing_configurations = [
    [5, 8, 8, 24, 0, 2, 0, 1, 0, 0.0, 0.0, 0, 1], # example for checkers
    [1, 6, 8, 48, 0, 1, 1, 0, 0, 0.0, 0.0, 3, 1],
]

# Same as was used to train the meta model
BRUTE_FORCE_SAMPLES = 250
BRUTE_FORCE_MAX_STEPS = 200

SEARCH_DEPTH = 2
MAX_TOTAL_MOVES = 200
MODEL_PLAYER = 1


def _resolve_default_model_path():
    flag_string = "".join(str(int(bool(flag))) for flag in model.DATASETS)
    return model.OUTPUT_DIR / f"model_{flag_string}.pt"


def _load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    net = model.HeuristicMetaRegressor(
        checkpoint["input_dim"],
        checkpoint["output_dim"],
        checkpoint["hidden_sizes"],
        checkpoint["dropout"],
    ).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net, checkpoint


def _normalize_inputs(x_values, checkpoint):
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    return (x_values - x_mean) / x_std

def _denormalize_outputs(y_values, checkpoint):
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)
    return y_values * y_std + y_mean


# Unflattens the model's output vector into a list of (mu, sigma) tuples for each heuristic
def _vector_to_heuristics(vector):
    values = np.asarray(vector, dtype=np.float64)
    if values.shape[-1] != NUM_HEURISTICS * 2:
        raise ValueError(f"Expected {NUM_HEURISTICS * 2} outputs, got {values.shape[-1]}")

    predicted = []
    for idx in range(NUM_HEURISTICS):
        mu = float(values[2 * idx])
        sigma = float(values[2 * idx + 1])
        if sigma < 1e-6:
            sigma = 1e-6
        predicted.append((mu, sigma))

    return predicted


def _build_game_from_config(config):
    if len(config) != len(model.FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(model.FEATURE_NAMES)} values, got {len(config)}"
        )

    (game_id, rows, cols, _num_pieces, max_pieces_per_player, _num_unique_pieces, _placement_game,_captures, _space_game, edge_ratio, inner_ratio, in_a_row, turns_per_block) = config

    game_id = int(game_id)
    rows = int(rows)
    cols = int(cols)
    num_turns = int(turns_per_block)
    max_pieces = None if int(max_pieces_per_player) == 0 else int(max_pieces_per_player)
    edge_ratio = float(edge_ratio)
    inner_ratio = float(inner_ratio)
    in_a_row = int(in_a_row)

    if game_id == 1:
        return TicTacToe(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            in_a_row=in_a_row,
            max_pieces_per_player=max_pieces,
            num_turns=num_turns,
        )
    if game_id == 2:
        return ConnectFour(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            in_a_row=in_a_row,
            max_pieces_per_player=max_pieces,
            num_turns=num_turns,
        )
    if game_id == 3:
        return Othello(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            max_pieces_per_player=max_pieces,
            num_turns=num_turns,
        )
    if game_id == 4:
        return Ataxx(
            rows=rows,
            cols=cols,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            max_pieces_per_player=max_pieces,
            num_turns=num_turns,
        )
    if game_id == 5:
        return Checkers(
            rows=rows,
            cols=cols,
            keep_pieces=True,
            edge_unplayable_ratio=edge_ratio,
            inner_unplayable_ratio=inner_ratio,
            max_pieces_per_player=max_pieces,
            num_turns=num_turns,
        )

    raise ValueError(f"Unknown game_id: {game_id}")


def _piece_winner(state):
    board = state.board
    player_one = int(np.sum(board == 1))
    player_two = int(np.sum(board == -1))
    if player_one > player_two:
        return 1
    if player_two > player_one:
        return -1
    return 0


def _determine_winner(game, state, ended_by_limit):
    if ended_by_limit:
        return 0

    if isinstance(game, TicTacToe):
        return int(game.check_winner(state))
    if isinstance(game, ConnectFour):
        if game.check_winner(state, 1):
            return 1
        if game.check_winner(state, -1):
            return -1
        return 0
    if isinstance(game, (Othello, Ataxx)):
        return _piece_winner(state)
    if isinstance(game, Checkers):
        return -state.player if not game.legal_moves(state) else _piece_winner(state)

    return 0


def _play_match(game, model_predicted, baseline_predicted):
    state = game.initial_state()
    moves_played = 0

    while not game.game_over(state) and moves_played < MAX_TOTAL_MOVES:
        predicted = model_predicted if state.player == MODEL_PLAYER else baseline_predicted
        move = choose_move(game, state, predicted, depth=SEARCH_DEPTH)

        if move is None:
            if game.allows_pass():
                state = game.make_move(state, None)
                moves_played += 1
                continue
            break

        state = game.make_move(state, move)
        moves_played += 1

    ended_by_limit = moves_played >= MAX_TOTAL_MOVES
    winner = _determine_winner(game, state, ended_by_limit)

    return winner, moves_played, ended_by_limit


def _format_winner(winner, model_player):
    if winner == 0:
        return "draw"
    if winner == model_player:
        return "model"
    if winner == -model_player:
        return "baseline"
    return "draw"


def main():
    if MODEL_PATH:
        model_path = Path(MODEL_PATH)
    else:
        model.train()
        model_path = _resolve_default_model_path()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not testing_configurations:
        raise ValueError("testing_configurations must contain at least one entry")

    x_values = np.asarray(testing_configurations, dtype=np.float32)
    if x_values.ndim != 2 or x_values.shape[1] != len(model.FEATURE_NAMES):
        raise ValueError("Each configuration must have 13 values")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, checkpoint = _load_model(model_path, device) # 'model' name is taken by the imported module!

    x_norm = _normalize_inputs(x_values, checkpoint)
    with torch.no_grad():
        preds_norm = net(torch.from_numpy(x_norm).to(device)).cpu().numpy()
    preds = _denormalize_outputs(preds_norm, checkpoint)

    for idx, (config, pred) in enumerate(zip(testing_configurations, preds), start=1):
        game = _build_game_from_config(config)
        baseline = calculate_label_online(
            game,
            n_samples=BRUTE_FORCE_SAMPLES,
            max_steps_per_game=BRUTE_FORCE_MAX_STEPS,
        )
        model_predicted = _vector_to_heuristics(pred)
        winner, moves_played, ended_by_limit = _play_match(game, model_predicted, baseline)

        winner_text = _format_winner(winner, MODEL_PLAYER)
        limit_text = " (max move limit)" if ended_by_limit else ""
        game_id = int(config[0])
        print(f"Config {idx} (game_id={game_id}): winner={winner_text}{limit_text}, moves={moves_played}")


if __name__ == "__main__":
    main()
