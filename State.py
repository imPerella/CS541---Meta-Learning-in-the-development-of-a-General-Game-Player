import random
import numpy as np
#We need varying state classes as each game has a different inital state.

UNPLAYABLE = 9


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
    def __init__(self, board=None, player=1, rows=3, cols=3, unplayable_positions=None):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
            apply_unplayable_positions(self.board, unplayable_positions)
        else:
            self.board = board

        self.player = player  # 1 (X) or -1 (O)

class ConnectFourState:
    def __init__(self, board=None, player=1, rows=6, cols=7, unplayable_positions=None):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
            apply_unplayable_positions(self.board, unplayable_positions)
        else:
            self.board = board

        self.player = player  # 1 or -1 

class OthelloState:
    def __init__(self, board=None, player=1, rows = 8, cols = 8, unplayable_positions=None):
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


class AtaxxState:
    def __init__(self, board=None, player=1, rows = 7, cols = 7, unplayable_positions=None):
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
    ):
        if board is None and keep_pieces:
            self.board = np.zeros((rows, cols), dtype=int)

            target_pieces_per_side = min(12, (rows * cols) // 4)

            top_count = 0
            for row in range(rows):
                for col in range(row % 2, cols, 2):
                    if top_count >= target_pieces_per_side:
                        break
                    self.board[row, col] = 1
                    top_count += 1
                if top_count >= target_pieces_per_side:
                    break

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
            for row in range(0, rows):
                if row < (rows-1)//2:
                    for col in range(row % 2, cols, 2):
                        self.board[row, col] = 1
                elif row > round((rows-1)/2):
                    for col in range(row % 2, cols, 2):
                        self.board[row, col] = -1
            apply_unplayable_positions(self.board, unplayable_positions)
        else:
            self.board = board
        self.player = player # 1 or -1
