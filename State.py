import numpy as np
#We need varying state classes as each game has a different inital state.

class TicTacToeState:
    def __init__(self, board=None, player=1, rows=3, cols=3):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
        else:
            self.board = board

        self.player = player  # 1 (X) or -1 (O)

class ConnectFourState:
    def __init__(self, board=None, player=1, rows=6, cols=7):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
        else:
            self.board = board

        self.player = player  # 1 or -1 

class OthelloState:
    def __init__(self, board=None, player=1, rows = 8, cols = 8):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
            middle_row, middle_col = rows//2, cols//2
            self.board[middle_row-1, middle_col-1] = -1
            self.board[middle_row - 1 , middle_col] = 1
            self.board[middle_row, middle_col-1] = 1
            self.board[middle_row, middle_col] = -1

        else:
            self.board = board

        self.player = player  # 1 or -1


class AtaxxState:
    def __init__(self, board=None, player=1, rows = 7, cols = 7):
        if board is None:
            self.board = np.zeros((rows, cols), dtype=int)
            self.board[0, 0] = 1
            self.board[0, cols-1] = -1
            self.board[rows-1, 0] = -1
            self.board[rows-1, cols-1] = 1

        else:
            self.board = board

        self.player = player  # 1 or -1

#change later for modularity
class CheckersState:
    def __init__(self, board=None, player=1, rows = 8, cols = 8, keep_pieces = True):
        if board is None and keep_pieces:
            self.board = np.zeros((rows, cols), dtype=int)

            self.board[0, 0] = 1
            self.board[0, 2] = 1
            self.board[0, 4] = 1
            self.board[0, 6] = 1
            self.board[1, 1] = 1
            self.board[1, 3] = 1
            self.board[1, 5] = 1
            self.board[1, 7] = 1
            self.board[2, 0] = 1
            self.board[2, 2] = 1
            self.board[2, 4] = 1
            self.board[2, 6] = 1

            self.board[7, 1] = -1
            self.board[7, 3] = -1
            self.board[7, 5] = -1
            self.board[7, 7] = -1
            self.board[6, 0] = -1
            self.board[6, 2] = -1
            self.board[6, 4] = -1
            self.board[6, 6] = -1
            self.board[5, 1] = -1
            self.board[5, 3] = -1
            self.board[5, 5] = -1
            self.board[5, 7] = -1
        
        elif board is None and not keep_pieces:
            self.board = np.zeros((rows, cols), dtype=int)
            for row in range(0, rows):
                if row < (rows-1)//2:
                    for col in range(row % 2, cols, 2):
                        self.board[row, col] = 1
                elif row > round((rows-1)/2):
                    for col in range(row % 2, cols, 2):
                        self.board[row, col] = -1
        else:
            self.board = board
        self.player = player # 1 or -1
