import numpy as np
from Game import Game
from State import AtaxxState
class Ataxx(Game):

    DIRECTIONS = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)
                  if not (dr == 0 and dc == 0)]

    def __init__(self, rows=7, cols=7, keep_pieces = True):
        self.rows = rows
        self.cols = cols
        self.keep_pieces = keep_pieces

    def initial_state(self):
        return AtaxxState(rows=self.rows, cols=self.cols, keep_pieces=self.keep_pieces)

    # ----------------------
    # Move Logic
    # ----------------------

    def on_board(self, r, c, rows, cols):
        return 0 <= r < rows and 0 <= c < cols

    def legal_moves(self, state):
        board = state.board
        player = state.player
        rows, cols = board.shape

        moves = []

        for r in range(rows):
            for c in range(cols):
                if board[r, c] == player:
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if dr == 0 and dc == 0:
                                continue

                            nr, nc = r + dr, c + dc

                            if (self.on_board(nr, nc, rows, cols) and
                                board[nr, nc] == 0):

                                move_type = "clone" if max(abs(dr), abs(dc)) == 1 else "jump"
                                moves.append(((r, c), (nr, nc), move_type))

        return moves if moves else None

    def make_move(self, state, move):
        if move == None:
            return AtaxxState(state.board.copy(), -state.player)
        
        (r1, c1), (r2, c2), move_type = move
        board = state.board.copy()
        player = state.player

        if move_type == "clone":
            board[r2, c2] = player
        else:  # jump
            board[r1, c1] = 0
            board[r2, c2] = player

        # Flip adjacent opponent pieces
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r2 + dr, c2 + dc
                if (0 <= nr < board.shape[0] and
                    0 <= nc < board.shape[1] and
                    board[nr, nc] == -player):
                    board[nr, nc] = player

        return AtaxxState(board, -player)

    def game_over(self, state):
        if self.legal_moves(state) != None:
            return False

        opponent_state = AtaxxState(state.board, -state.player)
        if self.legal_moves(opponent_state) != None:
            return False

        return True

    # ----------------------
    # Heuristics
    # ----------------------

    def control(self, state):
        board = state.board
        player = state.player

        player_tiles = np.sum(board == player)
        total_tiles = self.rows * self.cols

        return player_tiles / total_tiles

    def mobility(self, state):
        if self.legal_moves(state) != None:
            player_moves = len(self.legal_moves(state))
            total_squares = self.rows*self.cols
        else: 
            return 0
        
        return player_moves / total_squares

    def stability(self, state):
        # Stable = pieces surrounded by same color (hard to flip)
        board = state.board
        player = state.player
        rows, cols = board.shape

        stable = 0
        total = np.sum(board == player)

        if total == 0:
            return 0

        for r in range(rows):
            for c in range(cols):
                if board[r, c] == player:
                    stable_piece = True
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < rows and 0 <= nc < cols):
                                if board[nr, nc] == -player:
                                    stable_piece = False
                    if stable_piece:
                        stable += 1

        return stable / total

    def connectivity(self, state):
        board = state.board
        player = state.player
        visited = set()

        def dfs(r, c):
            stack = [(r, c)]
            size = 0

            while stack:
                x, y = stack.pop()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                size += 1

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nx, ny = x + dr, y + dc
                        if (0 <= nx < rows and 0 <= ny < cols and
                            board[nx, ny] == player and
                            (nx, ny) not in visited):
                            stack.append((nx, ny))

            return size

        rows, cols = board.shape
        positions = list(zip(*np.where(board == player)))
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

        player = state.player
        impactful = 0

        for move in legal:
            next_state = self.make_move(state, move)

            # Count flips caused by move
            before = np.sum(state.board == player)
            after = np.sum(next_state.board == player)

            if after - before >= 2:  # strong gain
                impactful += 1

        return impactful / len(legal)
