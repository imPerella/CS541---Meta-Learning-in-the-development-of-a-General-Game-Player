import numpy as np
from State import OthelloState, UNPLAYABLE, clone_piece_queues, sample_unplayable_positions
from Game import Game

class Othello(Game):

    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)
    ]

    def __init__(
        self,
        rows=8,
        cols=8,
        keep_pieces=True,
        edge_unplayable_ratio=0.0,
        inner_unplayable_ratio=0.0,
        max_pieces_per_player=None,
        num_turns=1,
    ):
        self.rows = rows
        self.cols = cols
        self.keep_pieces = keep_pieces
        self.edge_unplayable_ratio = edge_unplayable_ratio
        self.inner_unplayable_ratio = inner_unplayable_ratio
        self.num_turns = self._validate_num_turns(num_turns)
        if max_pieces_per_player is not None and int(max_pieces_per_player) < 1:
            raise ValueError("max_pieces_per_player must be >= 1 when provided")
        self.max_pieces_per_player = None if max_pieces_per_player is None else int(max_pieces_per_player)

        middle_row, middle_col = self.rows // 2, self.cols // 2
        forbidden_positions = {
            (middle_row - 1, middle_col - 1),
            (middle_row - 1, middle_col),
            (middle_row, middle_col - 1),
            (middle_row, middle_col),
        }
        self.unplayable_positions = sample_unplayable_positions(
            rows=self.rows,
            cols=self.cols,
            edge_unplayable_ratio=self.edge_unplayable_ratio,
            inner_unplayable_ratio=self.inner_unplayable_ratio,
            forbidden_positions=forbidden_positions,
        )

    def _remove_from_queue(self, queue, position):
        for idx, queued_position in enumerate(queue):
            if queued_position == position:
                queue.pop(idx)
                return

    def _enforce_piece_limit(self, board, queues, player):
        if self.max_pieces_per_player is None:
            return

        while len(queues[player]) > self.max_pieces_per_player:
            old_r, old_c = queues[player].pop(0)
            if board[old_r, old_c] == player:
                board[old_r, old_c] = 0

    def initial_state(self):
        return OthelloState(
            rows=self.rows,
            cols=self.cols,
            unplayable_positions=self.unplayable_positions,
            num_turns=self.num_turns,
        )

    def allows_pass(self):
        return True

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
                if self.get_flips(state, r, c):
                    moves.append((r, c))

        return moves if moves else [None]

    def make_move(self, state, move):
        if move is None:
            next_player, next_turns_remaining = self._next_turn(state.player, state.turns_remaining)
            return OthelloState(
                state.board.copy(),
                next_player,
                piece_queues=clone_piece_queues(state.piece_queues),
                num_turns=self.num_turns,
                turns_remaining=next_turns_remaining,
            )
        
        r, c = move
        board = state.board.copy()
        player = state.player
        opponent = -player
        queues = clone_piece_queues(state.piece_queues)

        flips = self.get_flips(state, r, c)

        if not flips:
            return state  # invalid move safeguard

        board[r, c] = player
        queues[player].append((r, c))
        for fr, fc in flips:
            board[fr, fc] = player
            self._remove_from_queue(queues[opponent], (fr, fc))
            queues[player].append((fr, fc))

        self._enforce_piece_limit(board, queues, player)
        next_player, next_turns_remaining = self._next_turn(player, state.turns_remaining)

        return OthelloState(
            board,
            next_player,
            piece_queues=queues,
            num_turns=self.num_turns,
            turns_remaining=next_turns_remaining,
        )

    def game_over(self, state):
        # Game ends when neither player has moves
        if self.legal_moves(state) != [None]:
            return False

        opponent = OthelloState(
            state.board,
            -state.player,
            piece_queues=clone_piece_queues(state.piece_queues),
            num_turns=self.num_turns,
        )
        if self.legal_moves(opponent) != [None]:
            return False

        return True

    # ----------------------
    # Heuristics
    # ----------------------

    def control(self, state):
        board = state.board
        player = state.player

        player_tiles = np.sum(board == player)
        total_tiles = np.sum(board != UNPLAYABLE)

        if total_tiles == 0:
            return 0

        return player_tiles / total_tiles

    def mobility(self, state):
        legal = [m for m in self.legal_moves(state) if m is not None]
        player_moves = len(legal)
        total_squares = np.sum(state.board != UNPLAYABLE)

        if total_squares == 0:
            return 0

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
        rows, cols = board.shape
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

        return max_size / len(positions) #len positions = total num pieces of player

    def tension(self, state):
        player = state.player
        legal = [m for m in self.legal_moves(state) if m is not None]

        if not legal:
            return 0

        opponent_state = OthelloState(state.board, -player, num_turns=self.num_turns)
        opponent_moves = [m for m in self.legal_moves(opponent_state) if m is not None]

        # Moves that reduce opponent mobility
        impactful = 0

        for move in legal:
            next_state = self.make_move(state, move)
            next_opponent = OthelloState(next_state.board, -player, num_turns=self.num_turns)
            next_opponent_moves = len(
                [m for m in self.legal_moves(next_opponent) if m is not None]
            )

            if next_opponent_moves < len(opponent_moves):
                impactful += 1

        return impactful / len(legal)
