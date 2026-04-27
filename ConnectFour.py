import numpy as np
from Game import Game
from State import ConnectFourState, UNPLAYABLE, clone_piece_queues, sample_unplayable_positions

class ConnectFour(Game):

    def __init__(
        self,
        rows=6,
        cols=7,
        edge_unplayable_ratio=0.0,
        inner_unplayable_ratio=0.0,
        in_a_row=4,
        max_pieces_per_player=None,
        num_turns=1,
    ):
        self.rows = rows
        self.cols = cols
        max_in_a_row = min(self.rows, self.cols)
        self.in_a_row = int(in_a_row)
        if self.in_a_row < 3 or self.in_a_row > max_in_a_row:
            raise ValueError("in_a_row must be between 3 and min(rows, cols)")
        self.num_turns = self._validate_num_turns(num_turns)
        if max_pieces_per_player is not None and int(max_pieces_per_player) < 1:
            raise ValueError("max_pieces_per_player must be >= 1 when provided")
        self.max_pieces_per_player = None if max_pieces_per_player is None else int(max_pieces_per_player)
        self.edge_unplayable_ratio = edge_unplayable_ratio
        self.inner_unplayable_ratio = inner_unplayable_ratio
        self.unplayable_positions = sample_unplayable_positions(
            rows=self.rows,
            cols=self.cols,
            edge_unplayable_ratio=self.edge_unplayable_ratio,
            inner_unplayable_ratio=self.inner_unplayable_ratio,
        )

    def _enforce_piece_limit(self, board, queues, player):
        if self.max_pieces_per_player is None:
            return

        while len(queues[player]) > self.max_pieces_per_player:
            old_r, old_c = queues[player].pop(0)
            if board[old_r, old_c] == player:
                board[old_r, old_c] = 0

    def initial_state(self):
        return ConnectFourState(
            rows=self.rows,
            cols=self.cols,
            unplayable_positions=self.unplayable_positions,
            num_turns=self.num_turns,
        )

    def legal_moves(self, state):
        # A move is a column where the top cell is empty
        cols = self.cols
        return [c for c in range(cols) if state.board[0, c] == 0]

    def make_move(self, state, move):
        col = move
        board = state.board.copy()
        queues = clone_piece_queues(state.piece_queues)
        drop_row = None

        # Drop piece to lowest empty row in that column
        for row in reversed(range(self.rows)):
            if board[row, col] == 0:
                board[row, col] = state.player
                drop_row = row
                break

        if drop_row is None:
            raise ValueError("Illegal move")

        queues[state.player].append((drop_row, col))
        self._enforce_piece_limit(board, queues, state.player)
        next_player, next_turns_remaining = self._next_turn(state.player, state.turns_remaining)

        return ConnectFourState(
            board,
            next_player,
            piece_queues=queues,
            num_turns=self.num_turns,
            turns_remaining=next_turns_remaining,
        )

    def game_over(self, state):
        return (
            self.check_winner(state) or
            len(self.legal_moves(state)) == 0
        )

    # ----------------------
    # Win Detection
    # ----------------------

    def check_winner(self, state, player=None):
        rows, cols = state.board.shape
        k = self.in_a_row

        players = [player] if player in (-1, 1) else [1, -1]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for p in players:
            for r in range(rows):
                for c in range(cols):
                    if state.board[r, c] != p:
                        continue

                    for dr, dc in directions:
                        end_r = r + (k - 1) * dr
                        end_c = c + (k - 1) * dc
                        if not (0 <= end_r < rows and 0 <= end_c < cols):
                            continue

                        if all(state.board[r + i * dr, c + i * dc] == p for i in range(k)):
                            return True

        return False

    # ----------------------
    # Heuristics
    # ----------------------

    def control(self, state):
        board = state.board
        total = np.sum(board != UNPLAYABLE)
        if total == 0:
            return 0
        controlled = 0

        for c in range(self.cols):
            controlled += np.sum(board[:, c] == state.player)

        return controlled / total

    def mobility(self, state):
        total_playable = np.sum(state.board != UNPLAYABLE)
        if total_playable == 0:
            return 0
        return len(self.legal_moves(state)) / total_playable

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
            rows, cols = board.shape

            while stack:
                x, y = stack.pop()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                size += 1

                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if (0 <= nx < rows and 0 <= ny < cols and
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
            if self.check_winner(next_state, state.player):
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
        k = self.in_a_row

        if k < 2:
            return False

        def count_window(window):
            return (
                np.count_nonzero(window == player) == (k - 1) and
                np.count_nonzero(window == 0) == 1
            )

        # Check all k-length windows
        for r in range(rows):
            for c in range(cols - k + 1):
                if count_window(board[r, c:c+k]):
                    return True

        for r in range(rows - k + 1):
            for c in range(cols):
                if count_window(board[r:r+k, c]):
                    return True

        for r in range(rows - k + 1):
            for c in range(cols - k + 1):
                if count_window([board[r+i, c+i] for i in range(k)]):
                    return True

        for r in range(k - 1, rows):
            for c in range(cols - k + 1):
                if count_window([board[r-i, c+i] for i in range(k)]):
                    return True

        return False
