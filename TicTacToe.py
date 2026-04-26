import numpy as np
from State import TicTacToeState, UNPLAYABLE, sample_unplayable_positions
from Game import Game

class TicTacToe(Game):
    
    def __init__(
        self,
        rows=3,
        cols=3,
        edge_unplayable_ratio=0.0,
        inner_unplayable_ratio=0.0,
        in_a_row=None,
    ):
        self.rows = rows
        self.cols = cols
        max_in_a_row = min(self.rows, self.cols)
        if in_a_row is None:
            self.in_a_row = max_in_a_row
        else:
            self.in_a_row = int(in_a_row)
            if self.in_a_row < 3 or self.in_a_row > max_in_a_row:
                raise ValueError("in_a_row must be between 3 and min(rows, cols)")
        self.edge_unplayable_ratio = edge_unplayable_ratio
        self.inner_unplayable_ratio = inner_unplayable_ratio
        self.unplayable_positions = sample_unplayable_positions(
            rows=self.rows,
            cols=self.cols,
            edge_unplayable_ratio=self.edge_unplayable_ratio,
            inner_unplayable_ratio=self.inner_unplayable_ratio,
        )

    def initial_state(self):
        return TicTacToeState(
            rows=self.rows,
            cols=self.cols,
            unplayable_positions=self.unplayable_positions,
        )

    def legal_moves(self, state):
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                if state.board[r, c] == 0:
                    moves.append((r, c))
        return moves

    def make_move(self, state, move):
        r, c = move
        if state.board[r, c] != 0:
            raise ValueError("Illegal move")
        new_board = state.board.copy()
        new_board[r, c] = state.player

        return TicTacToeState(
            board=new_board,
            player=-state.player
        )

    def check_winner(self, state):
        if self.is_winning_state(state, 1):
            return 1
        if self.is_winning_state(state, -1):
            return -1
        return 0

    def game_over(self, state):

        # someone won
        if self.check_winner(state) != 0:
            return True

        # board full
        if len(self.legal_moves(state)) == 0:
            return True

        return False

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def control(self, state):
        player = state.player
        controlled_squares = np.sum(state.board == player)
        total_squares = np.sum(state.board != UNPLAYABLE)

        if total_squares == 0:
            return 0

        return controlled_squares / total_squares

    def mobility(self, state):
        total_squares = np.sum(state.board != UNPLAYABLE)
        if total_squares == 0:
            return 0
        return len(self.legal_moves(state)) / total_squares
    
     #Tic-Tac-Toe has no movable pieces.
    def stability(self, state):
        player = state.player
        total = np.sum(state.board == player)
        if total == 0:
            return 0
        else:
            return 1.0
    
    def connectivity(self, state):
        # Count largest connected component (4-directional)
        board = state.board
        rows, cols = board.shape
        visited = set()
        max_size = 0

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
                    if (0 <= nx < rows and 0 <= ny < cols and
                        board[nx, ny] == state.player and
                        (nx, ny) not in visited):
                        stack.append((nx, ny))
            return size

        player_positions = list(zip(*np.where(board == state.player)))

        for pos in player_positions:
            if pos not in visited:
                max_size = max(max_size, dfs(*pos))

        total_player_pieces = len(player_positions)
        if total_player_pieces == 0:
            return 0

        return max_size / total_player_pieces
    
    def tension(self, state):
        legal = self.legal_moves(state)
        if not legal:
            return 0

        threatening_moves = 0

        for move in legal:
            new_state = self.make_move(state, move)

            # check if move creates a win
            if self.is_winning_state(new_state, state.player):
                threatening_moves += 1

        return threatening_moves / len(legal)

    # helper
    def is_winning_state(self, state, player):
        board = state.board
        rows, cols = board.shape
        k = self.in_a_row

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(rows):
            for c in range(cols):
                if board[r, c] != player:
                    continue

                for dr, dc in directions:
                    end_r = r + (k - 1) * dr
                    end_c = c + (k - 1) * dc
                    if not (0 <= end_r < rows and 0 <= end_c < cols):
                        continue

                    if all(board[r + i * dr, c + i * dc] == player for i in range(k)):
                        return True
        return False

