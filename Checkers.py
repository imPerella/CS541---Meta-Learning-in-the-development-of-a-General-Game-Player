import numpy as np
from State import CheckersState, UNPLAYABLE, sample_unplayable_positions
from Game import Game
class Checkers(Game):

    def __init__(
        self,
        rows=8,
        cols=8,
        keep_pieces = True,
        edge_unplayable_ratio=0.0,
        inner_unplayable_ratio=0.0,
    ):
        self.rows = rows
        self.cols = cols
        self.keep_pieces = keep_pieces
        self.edge_unplayable_ratio = edge_unplayable_ratio
        self.inner_unplayable_ratio = inner_unplayable_ratio

        base_state = CheckersState(rows=self.rows, cols=self.cols, keep_pieces=self.keep_pieces)
        forbidden_positions = set(zip(*np.where(base_state.board != 0)))

        self.unplayable_positions = sample_unplayable_positions(
            rows=self.rows,
            cols=self.cols,
            edge_unplayable_ratio=self.edge_unplayable_ratio,
            inner_unplayable_ratio=self.inner_unplayable_ratio,
            forbidden_positions=forbidden_positions,
        )

    def initial_state(self):
        return CheckersState(
            rows=self.rows,
            cols=self.cols,
            keep_pieces=self.keep_pieces,
            unplayable_positions=self.unplayable_positions,
        )

    # ----------------------
    # Helpers
    # ----------------------

    def on_board(self, r, c, rows, cols):
        return 0 <= r < rows and 0 <= c < cols

    def is_king(self, piece):
        return abs(piece) == 2

    def directions(self, piece):
        # Men move forward only; kings both ways
        if piece == 1:
            return [(1, -1), (1, 1)]
        elif piece == -1:
            return [(-1, -1), (-1, 1)]
        else:  # kings
            return [(-1,-1),(-1,1),(1,-1),(1,1)]

    # ----------------------
    # Move Generation
    # ----------------------

    def legal_moves(self, state):
        board = state.board
        player = state.player
        rows, cols = board.shape

        captures = []
        moves = []

        for r in range(rows):
            for c in range(cols):
                piece = board[r, c]
                if piece * player > 0:  # player's piece

                    caps = self.get_captures(board, r, c)
                    if caps:
                        captures.extend(caps)
                    else:
                        moves.extend(self.get_simple_moves(board, r, c))

        # Forced capture rule
        return captures if captures else moves

    def get_simple_moves(self, board, r, c):
        piece = board[r, c]
        rows, cols = board.shape
        result = []

        for dr, dc in self.directions(piece):
            nr, nc = r + dr, c + dc
            if self.on_board(nr, nc, rows, cols) and board[nr, nc] == 0:
                result.append([(r, c), (nr, nc)])

        return result

    def get_captures(self, board, r, c):
        piece = board[r, c]
        rows, cols = board.shape
        captures = []

        def dfs(path_board, r, c, path):
            found = False

            for dr, dc in self.directions(path_board[r, c]):
                mr, mc = r + dr, c + dc
                nr, nc = r + 2*dr, c + 2*dc

                if (self.on_board(nr, nc, rows, cols) and
                    path_board[mr, mc] * path_board[r, c] < 0 and
                    path_board[nr, nc] == 0):

                    new_board = path_board.copy()
                    new_board[nr, nc] = new_board[r, c]
                    new_board[r, c] = 0
                    new_board[mr, mc] = 0

                    dfs(new_board, nr, nc, path + [(nr, nc)])
                    found = True

            if not found and len(path) > 1:
                captures.append(path)

        dfs(board, r, c, [(r, c)])
        return captures

    # ----------------------
    # Apply Move
    # ----------------------

    def make_move(self, state, move):
        board = state.board.copy()
        player = state.player

        for i in range(len(move) - 1):
            r1, c1 = move[i]
            r2, c2 = move[i+1]

            # Move piece
            board[r2, c2] = board[r1, c1]
            board[r1, c1] = 0

            # Capture
            if abs(r2 - r1) == 2:
                mr, mc = (r1 + r2)//2, (c1 + c2)//2
                board[mr, mc] = 0

        # Promotion
        rows = board.shape[0]
        for c in range(board.shape[1]):
            if board[rows-1, c] == 1:
                board[rows-1, c] = 2
            if board[0, c] == -1:
                board[0, c] = -2

        return CheckersState(board, -player)

    def game_over(self, state):
        if len(self.legal_moves(state)) == 0:
            return True
        return False

    # ----------------------
    # Heuristics
    # ----------------------

    def control(self, state):
        board = state.board
        player = state.player

        # Weighted: kings more valuable
        player_score = np.sum((board == player) * 1) + np.sum((board == 2*player) * 2)
        rows, cols = board.shape
        rr, cc = np.indices((rows, cols))
        dark_squares = (rr + cc) % 2 == 0
        total_squares = np.sum(dark_squares & (board != UNPLAYABLE))

        if total_squares == 0:
            return 0

        return player_score / total_squares

    def mobility(self, state):
        player_moves = len(self.legal_moves(state))
        opponent_moves = len(self.legal_moves(CheckersState(state.board, -state.player)))

        rows, cols = state.board.shape
        rr, cc = np.indices((rows, cols))
        dark_squares = (rr + cc) % 2 == 0
        total_squares = np.sum(dark_squares & (state.board != UNPLAYABLE))

        if total_squares == 0:
            return 0

        return player_moves / total_squares

    def stability(self, state):
        # Back-row pieces are more stable
        board = state.board
        player = state.player
        rows = board.shape[0]

        stable = 0
        total = np.sum(board * player > 0)

        if total == 0:
            return 0

        for c in range(board.shape[1]):
            if board[rows-1, c] == player:
                stable += 1

        return stable / total

    def connectivity(self, state):
        board = state.board
        player = state.player
        visited = set()

        rows, cols = board.shape

        def dfs(r, c):
            stack = [(r, c)]
            size = 0

            while stack:
                x, y = stack.pop()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                size += 1

                for dr in [-1, 1]:
                    for dc in [-1, 1]:
                        nx, ny = x + dr, y + dc
                        if (0 <= nx < rows and 0 <= ny < cols and
                            board[nx, ny] * player > 0 and
                            (nx, ny) not in visited):
                            stack.append((nx, ny))

            return size

        positions = list(zip(*np.where(board * player > 0)))
        max_size = 0

        for pos in positions:
            if pos not in visited:
                max_size = max(max_size, dfs(*pos))

        if len(positions) == 0:
            return 0

        return max_size / len(positions)

    def tension(self, state):
        legal = self.legal_moves(state)
        if not legal:
            return 0

        # Tension = proportion of capture moves
        capture_moves = 0
        for m in legal:
            if any(abs(m[i+1][0] - m[i][0]) == 2 for i in range(len(m) - 1)):
                capture_moves += 1

        return capture_moves / len(legal)
