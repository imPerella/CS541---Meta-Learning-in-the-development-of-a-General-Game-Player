import numpy as np
from State import TicTacToeState
from Game import Game

class TicTacToe(Game):
    
    def __init__(self, rows=3, cols=3):
        self.rows = rows
        self.cols = cols

    def initial_state(self):
        return TicTacToeState(rows=self.rows, cols=self.cols)

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

        board = state.board

        # rows
        for r in range(self.rows):
            row_sum = np.sum(board[r])
            if row_sum == self.cols:
                return 1
            if row_sum == -self.cols:
                return -1

        # columns
        for c in range(self.cols):
            col_sum = np.sum(board[:, c])
            if col_sum == self.rows:
                return 1
            if col_sum == -self.rows:
                return -1

        # main diagonal
        diag = np.sum(np.diag(board))
        if diag == self.rows:
            return 1
        if diag == -self.rows:
            return -1

        # anti diagonal
        anti = np.sum(np.diag(np.fliplr(board)))
        if anti == self.rows:
            return 1
        if anti == -self.rows:
            return -1
        
        return 0

    def game_over(self, state):

        # someone won
        if self.check_winner(state) != 0:
            return True

        # board full
        if not np.any(state.board == 0):
            return True

        return False

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def control(self, state):
        player = state.player
        controlled_squares = np.sum(state.board == player)
        total_squares = self.rows * self.cols
        return controlled_squares / total_squares

    def mobility(self, state):
        total_squares = self.rows * self.cols
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
                    if (0 <= nx < 3 and 0 <= ny < 3 and
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

        lines = []
        lines.extend(board)
        lines.extend(board.T)
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))

        for line in lines:
            if sum(line) == 3 * player:
                return True
        return False
