import numpy as np
from Game import Game
from State import ConnectFourState

class ConnectFour(Game):

    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols

    def initial_state(self):
        return ConnectFourState(rows=self.rows, cols=self.cols)

    def legal_moves(self, state):
        # A move is a column where the top cell is empty
        cols = self.cols
        return [c for c in range(cols) if state.board[0, c] == 0]

    def make_move(self, state, move):
        col = move
        board = state.board.copy()

        # Drop piece to lowest empty row in that column
        for row in reversed(range(self.rows)):
            if board[row, col] == 0:
                board[row, col] = state.player
                break

        return ConnectFourState(board, -state.player)

    def game_over(self, state):
        return (
            self.check_winner(state) or
            len(self.legal_moves(state)) == 0
        )

    # ----------------------
    # Win Detection
    # ----------------------

    def check_winner(self, state):
        rows, cols = state.board.shape

        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if all(state.board[r, c+i] == state.player for i in range(4)):
                    return True

        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if all(state.board[r+i, c] == state.player for i in range(4)):
                    return True

        # Diagonal /
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(state.board[r+i, c+i] == state.player for i in range(4)):
                    return True

        # Diagonal \
        for r in range(3, rows):
            for c in range(cols - 3):
                if all(state.board[r-i, c+i] == state.player for i in range(4)):
                    return True

        return False

    # ----------------------
    # Heuristics
    # ----------------------

    def control(self, state):
        board = state.board
        total = self.cols * self.rows
        controlled = 0

        for c in range(self.cols):
            controlled += np.sum(board[:, c] == state.player)

        return controlled / total

    def mobility(self, state):
        return len(self.legal_moves(state)) / self.cols*self.rows

    def stability(self, state):
        # Stable = pieces that cannot be captured/moved
        board = state.board
        player = state.player

        total = np.sum(board == player)

        if total == 0:
            return 0
        
        else:
            return 1.0

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

                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if (0 <= nx < 6 and 0 <= ny < 7 and
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

        return max_size / len(positions)

    def tension(self, state):
        legal = self.legal_moves(state)
        if not legal:
            return 0

        threats = 0

        for move in legal:
            next_state = self.make_move(state, move)

            # Immediate win
            if self.check_winner(next_state):
                threats += 1
                continue

            # Check if move creates a 3-in-a-row threat
            if self.creates_threat(next_state.board, state.player):
                threats += 1

        return threats / len(legal)

    # ----------------------
    # Threat Detection
    # ----------------------

    def creates_threat(self, board, player):
        rows, cols = board.shape

        def count_window(window):
            return (
                np.count_nonzero(window == player) == 3 and
                np.count_nonzero(window == 0) == 1
            )

        # Check all 4-length windows
        for r in range(rows):
            for c in range(cols - 3):
                if count_window(board[r, c:c+4]):
                    return True

        for r in range(rows - 3):
            for c in range(cols):
                if count_window(board[r:r+4, c]):
                    return True

        for r in range(rows - 3):
            for c in range(cols - 3):
                if count_window([board[r+i, c+i] for i in range(4)]):
                    return True

        for r in range(3, rows):
            for c in range(cols - 3):
                if count_window([board[r-i, c+i] for i in range(4)]):
                    return True

        return False
