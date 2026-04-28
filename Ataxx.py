import numpy as np
from Game import Game
from State import AtaxxState, UNPLAYABLE, clone_piece_queues, sample_unplayable_positions
class Ataxx(Game):

    DIRECTIONS = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)
                  if not (dr == 0 and dc == 0)]

    def __init__(
        self,
        rows=7,
        cols=7,
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
        forbidden_positions = {
            (0, 0),
            (0, self.cols - 1),
            (self.rows - 1, 0),
            (self.rows - 1, self.cols - 1),
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

    def _move_position_in_queue(self, queue, from_position, to_position):
        for idx, queued_position in enumerate(queue):
            if queued_position == from_position:
                queue[idx] = to_position
                return
        queue.append(to_position)

    def _enforce_piece_limit(self, board, queues, player):
        if self.max_pieces_per_player is None:
            return

        while len(queues[player]) > self.max_pieces_per_player:
            old_r, old_c = queues[player].pop(0)
            if board[old_r, old_c] == player:
                board[old_r, old_c] = 0

    def initial_state(self):
        return AtaxxState(
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
            next_player, next_turns_remaining = self._next_turn(state.player, state.turns_remaining)
            return AtaxxState(
                state.board.copy(),
                next_player,
                piece_queues=clone_piece_queues(state.piece_queues),
                num_turns=self.num_turns,
                turns_remaining=next_turns_remaining,
            )
        
        (r1, c1), (r2, c2), move_type = move
        board = state.board.copy()
        player = state.player
        opponent = -player
        queues = clone_piece_queues(state.piece_queues)

        if move_type == "clone":
            board[r2, c2] = player
            queues[player].append((r2, c2))
        else:  # jump
            board[r1, c1] = 0
            board[r2, c2] = player
            self._move_position_in_queue(queues[player], (r1, c1), (r2, c2))

        # Flip adjacent opponent pieces
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r2 + dr, c2 + dc
                if (0 <= nr < board.shape[0] and
                    0 <= nc < board.shape[1] and
                    board[nr, nc] == opponent):
                    board[nr, nc] = player
                    self._remove_from_queue(queues[opponent], (nr, nc))
                    queues[player].append((nr, nc))

        self._enforce_piece_limit(board, queues, player)
        next_player, next_turns_remaining = self._next_turn(player, state.turns_remaining)

        return AtaxxState(
            board,
            next_player,
            piece_queues=queues,
            num_turns=self.num_turns,
            turns_remaining=next_turns_remaining,
        )

    def game_over(self, state):
        if self.legal_moves(state) != None:
            return False

        opponent_state = AtaxxState(
            state.board,
            -state.player,
            piece_queues=clone_piece_queues(state.piece_queues),
            num_turns=self.num_turns,
        )
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
        total_tiles = np.sum(board != UNPLAYABLE)

        if total_tiles == 0:
            return 0

        return player_tiles / total_tiles

    def mobility(self, state):
        if self.legal_moves(state) != None:
            player_moves = len(self.legal_moves(state))
            total_squares = np.sum(state.board != UNPLAYABLE)
            if total_squares == 0:
                return 0
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
