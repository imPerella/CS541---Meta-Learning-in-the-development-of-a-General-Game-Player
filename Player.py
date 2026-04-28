import numpy as np
import time

def sample_weights(predicted):
    weights = []
    for (mu, sigma) in predicted:
        weights.append(np.random.normal(mu, sigma))
    return weights


def evaluate_state(game, state, root_player, weights):
    current_player = state.player
    state.player = root_player

    score = (
        weights[0] * game.control(state)
        + weights[1] * game.mobility(state)
        + weights[2] * game.stability(state)
        + weights[3] * game.connectivity(state)
        + weights[4] * game.tension(state)
    )

    state.player = current_player
    return score



# Minimax + Alpha Beta Pruning because complex games will take forever with minimax;
def alphabeta(game, state, depth, alpha, beta, weights, root_player, start_time, time_limit = 1.0):
    if time.time() - start_time > time_limit:
        raise TimeoutError
    #bool
    maximizing_player = state.player == root_player 
    
    #leaf node of game tree
    if depth == 0 or game.game_over(state):
        return evaluate_state(
            game,
            state,
            root_player,
            weights,
        )

    #get all legal moves
    moves = game.legal_moves(state)

    # Pass turn if the game allows it (Othello and Ataxx handle None move inputs, as having no legal moves in these games is not considered a loss, you can skip your turn)
    # If not Othello or Ataxx, treat the lack of legal moves as a terminal state and evaluate it at that point
    if moves is None or moves == [None]:
        if not game.allows_pass():
            return evaluate_state(
                game,
                state,
                root_player,
                weights,
            )

        next_state = game.make_move(state, None)

        return alphabeta(
            game,
            next_state,
            depth - 1,
            alpha,
            beta,
            weights,
            root_player=root_player,
            start_time=start_time,
            time_limit=time_limit,
        )

    if len(moves) == 0: # this is another "no legal moves" case but want to run for all games (given it's true)
        return evaluate_state(
            game,
            state,
            root_player,
            weights,
        )
    
    if maximizing_player:

        value = -float("inf")

        for move in moves:

            next_state = game.make_move(state, move)

            score = alphabeta(
                game,
                next_state,
                depth - 1,
                alpha,
                beta,
                weights,
                root_player=root_player,
                start_time=start_time,
                time_limit=time_limit,
            )

            value = max(value, score)

            alpha = max(alpha, value)

            if beta <= alpha:
                break

        return value

    # Minimizing player

    else:

        value = float("inf")

        for move in moves:

            next_state = game.make_move(state, move)

            score = alphabeta(
                game,
                next_state,
                depth - 1,
                alpha,
                beta,
                weights,
                root_player=root_player,
                start_time=start_time,
                time_limit=time_limit,
            )

            value = min(value, score)

            beta = min(beta, value)

            if beta <= alpha:
                break

        return value


def choose_move(game, state, predicted, depth = 10):
    #start the timer
    start_time = time.time()

    #necessary as eval doesn't inherently have fixed perspective
    root_player = state.player

    #unsure if we should be sampling every move but we can for now
    weights = sample_weights(predicted)

    moves = game.legal_moves(state)

    if moves is None or moves == [None] or len(moves) == 0:
        return None

    best_move = None
    best_value = -float("inf")

    alpha = -float("inf")
    beta = float("inf")

    for move in moves:

        next_state = game.make_move(state, move)

        try:
            value = alphabeta(
                game,
                next_state,
                depth - 1,
                alpha,
                beta,
                weights,
                root_player=root_player,
                start_time=start_time,
            )
        except TimeoutError:
            break

        if value > best_value:

            best_value = value
            best_move = move

        alpha = max(alpha, best_value)

    return best_move