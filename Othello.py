import numpy as np
from State import OthelloState
from Game import Game

class Othello(Game):

    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)
    ]

    def __init__(self, rows=8, cols=8, keep_pieces = True):
        self.rows = rows
        self.cols = cols
        self.keep_pieces = keep_pieces

    def initial_state(self):
        return OthelloState(rows=self.rows, cols=self.cols, keep_pieces=self.keep_pieces)

    # ----------------------
    # Move Logic
    # ----------------------

    def on_board(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_flips(self, state, r, c):
        if state.board[r, c] != 0:
            return []

        opponent = -state.player
        flips = []

        for dr, dc in self.DIRECTIONS:
            path = []
            nr, nc = r + dr, c + dc

            while self.on_board(nr, nc) and state.board[nr, nc] == opponent:
                path.append((nr, nc))
                nr += dr
                nc += dc

            if (self.on_board(nr, nc) and
                state.board[nr, nc] == state.player and
                len(path) > 0):
                flips.extend(path)

        return flips

    def legal_moves(self, state):
        board = state.board
        player = state.player
        moves = []

        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if self.get_flips(board, r, c, player):
                    moves.append((r, c))

        return moves if moves else [None]

    def make_move(self, state, move):
        if move is None:
            return OthelloState(state.board.copy(), -state.player)
        
        r, c = move
        board = state.board.copy()
        player = state.player

        flips = self.get_flips(board, r, c, player)

        if not flips:
            return state  # invalid move safeguard

        board[r, c] = player
        for fr, fc in flips:
            board[fr, fc] = player

        return OthelloState(board, -player)

    def game_over(self, state):
        # Game ends when neither player has moves
        if self.legal_moves(state) != [None]:
            return False

        other = OthelloState(state.board, -state.player)
        if self.legal_moves(other) != [None]:
            return False

        return True

    # ----------------------
    # Heuristics
    # ----------------------

    def control(self, state):
        board = state.board
        player = state.player

        player_tiles = np.sum(board == player)
        total_tiles = self.cols*self.rows

        return player_tiles / total_tiles

    def mobility(self, state):
        player_moves = len(self.legal_moves(state))
        total_squares = self.cols *self.rows

        return player_moves / total_squares

    def stability(self, state):
        # Approximation: corners + edges connected to corners
        board = state.board
        player = state.player
        rows, cols = board.shape

        stable = 0
        total = np.sum(board == player)

        if total == 0:
            return 0

        corners = [(0,0),(0,cols-1),(rows-1,0),(rows-1,cols-1)]

        for r, c in corners:
            if board[r, c] == player:
                stable += 1

        # Edge stability approximation
        for i in range(rows):
            for j in range(cols):
                if board[i, j] == player:
                    if i in [0, rows-1] or j in [0, cols-1]:
                        stable += 0.5

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

                for dx, dy in self.DIRECTIONS:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < 8 and 0 <= ny < 8 and
                        board[nx, ny] == player and
                        (nx, ny) not in visited):
                        stack.append((nx, ny))

            return size

        positions = list(zip(*np.where(board == player)))
        max_size = 0

        for pos in positions:
            if pos not in visited:
                max_size = max(max_size, dfs(*pos))

        if len(positions) == 0:
            return 0

        return max_size / len(positions) #len positions = total num pieces of player

    def tension(self, state):
        player = state.player
        legal = self.legal_moves(state)

        if not legal:
            return 0

        opponent = OthelloState(state.board, -player)
        opponent_moves = set(self.legal_moves(opponent))

        # Moves that reduce opponent mobility
        impactful = 0

        for move in legal:
            next_state = self.make_move(state, move)
            next_opponent_moves = len(self.legal_moves(
                OthelloState(next_state.board, -player)
            ))

            if next_opponent_moves < len(opponent_moves):
                impactful += 1

        return impactful / len(legal)