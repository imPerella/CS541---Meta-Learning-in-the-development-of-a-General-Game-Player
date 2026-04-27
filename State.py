import random
import numpy as np
#We need varying state classes as each game has a different inital state.

UNPLAYABLE = 9


def normalize_num_turns(num_turns):
    normalized = int(num_turns)
    if normalized == 0:
        raise ValueError("num_turns must be non-zero")
    return normalized


def turn_quota_for_player(num_turns, player):
    if num_turns > 0:
        return num_turns if player == 1 else 1
    return abs(num_turns) if player == -1 else 1


def resolve_turns_remaining(num_turns, player, turns_remaining=None):
    if turns_remaining is None:
        return turn_quota_for_player(num_turns, player)

    normalized_turns_remaining = int(turns_remaining)
    if normalized_turns_remaining < 1:
        raise ValueError("turns_remaining must be >= 1")
    return normalized_turns_remaining


def clone_piece_queues(piece_queues):
    if piece_queues is None:
        return {1: [], -1: []}

    return {
        1: list(piece_queues.get(1, [])),
        -1: list(piece_queues.get(-1, [])),
    }


def build_piece_queues(board):
    queues = {1: [], -1: []}
    rows, cols = board.shape

    for r in range(rows):
        for c in range(cols):
            value = board[r, c]
            if value == UNPLAYABLE or value == 0:
                continue
            if value > 0:
                queues[1].append((r, c))
            else:
                queues[-1].append((r, c))

    return queues


def _split_edge_and_inner_cells(rows, cols, forbidden_positions=None):
    forbidden = forbidden_positions or set()
    edge = []
    inner = []

    for r in range(rows):
        for c in range(cols):
            if (r, c) in forbidden:
                continue

            if r in (0, rows - 1) or c in (0, cols - 1):
                edge.append((r, c))
            else:
                inner.append((r, c))

    return edge, inner


def sample_unplayable_positions(
    rows,
    cols,
    edge_unplayable_ratio=0.0,
    inner_unplayable_ratio=0.0,
    forbidden_positions=None,
):
    edge, inner = _split_edge_and_inner_cells(rows, cols, forbidden_positions)

    edge_count = min(len(edge), int(round(max(0.0, edge_unplayable_ratio) * len(edge))))
    inner_count = min(len(inner), int(round(max(0.0, inner_unplayable_ratio) * len(inner))))

    blocked = set()
    if edge_count > 0:
        blocked.update(random.sample(edge, edge_count))
    if inner_count > 0:
        blocked.update(random.sample(inner, inner_count))

    return blocked


def apply_unplayable_positions(board, unplayable_positions=None):
    if not unplayable_positions:
        return board

    for r, c in unplayable_positions:
        if board[r, c] == 0:
            board[r, c] = UNPLAYABLE

    return board

class TicTacToeState:
    def __init__(self, board=None, player=1, rows=3, cols=3, unplayable_positions=None, piece_queues=None, num_turns=1, turns_remaining=None):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
            apply_unplayable_positions(self.board, unplayable_positions)
        else:
            self.board = board

        self.player = player  # 1 (X) or -1 (O)
        self.num_turns = normalize_num_turns(num_turns)
        self.turns_remaining = resolve_turns_remaining(self.num_turns, self.player, turns_remaining)
        self.piece_queues = clone_piece_queues(piece_queues) if piece_queues is not None else build_piece_queues(self.board)

class ConnectFourState:
    def __init__(self, board=None, player=1, rows=6, cols=7, unplayable_positions=None, piece_queues=None, num_turns=1, turns_remaining=None):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
            apply_unplayable_positions(self.board, unplayable_positions)
        else:
            self.board = board

        self.player = player  # 1 or -1 
        self.num_turns = normalize_num_turns(num_turns)
        self.turns_remaining = resolve_turns_remaining(self.num_turns, self.player, turns_remaining)
        self.piece_queues = clone_piece_queues(piece_queues) if piece_queues is not None else build_piece_queues(self.board)

class OthelloState:
    def __init__(self, board=None, player=1, rows = 8, cols = 8, unplayable_positions=None, piece_queues=None, num_turns=1, turns_remaining=None):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
            middle_row, middle_col = rows//2, cols//2
            self.board[middle_row-1, middle_col-1] = -1
            self.board[middle_row - 1 , middle_col] = 1
            self.board[middle_row, middle_col-1] = 1
            self.board[middle_row, middle_col] = -1
            apply_unplayable_positions(self.board, unplayable_positions)

        else:
            self.board = board

        self.player = player  # 1 or -1
        self.num_turns = normalize_num_turns(num_turns)
        self.turns_remaining = resolve_turns_remaining(self.num_turns, self.player, turns_remaining)
        self.piece_queues = clone_piece_queues(piece_queues) if piece_queues is not None else build_piece_queues(self.board)


class AtaxxState:
    def __init__(self, board=None, player=1, rows = 7, cols = 7, unplayable_positions=None, piece_queues=None, num_turns=1, turns_remaining=None):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
            self.board[0, 0] = 1
            self.board[0, cols-1] = -1
            self.board[rows-1, 0] = -1
            self.board[rows-1, cols-1] = 1
            apply_unplayable_positions(self.board, unplayable_positions)

        else:
            self.board = board

        self.player = player  # 1 or -1
        self.num_turns = normalize_num_turns(num_turns)
        self.turns_remaining = resolve_turns_remaining(self.num_turns, self.player, turns_remaining)
        self.piece_queues = clone_piece_queues(piece_queues) if piece_queues is not None else build_piece_queues(self.board)

#change later for modularity
class CheckersState:
    def __init__(
        self,
        board=None,
        player=1,
        rows = 8,
        cols = 8,
        keep_pieces = True,
        unplayable_positions=None,
        max_pieces_per_player=None,
        num_turns=1,
        turns_remaining=None,
    ):
        if board is None and keep_pieces:
            self.board = np.zeros((rows, cols), dtype=int)

            target_pieces_per_side = min(12, (rows * cols) // 4)
            if max_pieces_per_player is not None:
                target_pieces_per_side = min(target_pieces_per_side, int(max_pieces_per_player))

            # Player 1 pieces
            top_count = 0
            for row in range(rows):
                for col in range(row % 2, cols, 2):
                    if top_count >= target_pieces_per_side:
                        break
                    self.board[row, col] = 1
                    top_count += 1
                if top_count >= target_pieces_per_side:
                    break

            # Player 2 (-1) pieces
            bottom_count = 0
            for row in range(rows - 1, -1, -1):
                for col in range(row % 2, cols, 2):
                    if bottom_count >= target_pieces_per_side:
                        break
                    if self.board[row, col] == 0:
                        self.board[row, col] = -1
                        bottom_count += 1
                if bottom_count >= target_pieces_per_side:
                    break
            apply_unplayable_positions(self.board, unplayable_positions)
        
        elif board is None and not keep_pieces:
            self.board = np.zeros((rows, cols), dtype=int)
            max_pieces = None if max_pieces_per_player is None else int(max_pieces_per_player)
            top_count = 0
            bottom_count = 0
            for row in range(0, rows):
                if row < (rows-1)//2:
                    for col in range(row % 2, cols, 2):
                        if max_pieces is not None and top_count >= max_pieces:
                            break
                        self.board[row, col] = 1
                        top_count += 1
                elif row > round((rows-1)/2):
                    for col in range(row % 2, cols, 2):
                        if max_pieces is not None and bottom_count >= max_pieces:
                            break
                        self.board[row, col] = -1
                        bottom_count += 1
            apply_unplayable_positions(self.board, unplayable_positions)
        else:
            self.board = board
        self.player = player # 1 or -1
        self.num_turns = normalize_num_turns(num_turns)
        self.turns_remaining = resolve_turns_remaining(self.num_turns, self.player, turns_remaining)
        self.max_pieces_per_player = max_pieces_per_player
