import numpy as np
from scipy.signal import convolve2d

class Board(object):

    def __init__(self):
        self.grid_ = np.zeros((6, 7), dtype=np.int8)
    
    def getMoves(self):
        moves = []
        for j in range(self.grid_.shape[1]):
            if self.grid_[0, j] == 0:
                moves.append(j)
        return moves
    
    def getSuccessor(self, player, move):
        if player not in {1, -1}:
            raise("Invalid player")
        if move >= self.grid_.shape[1]:
            raise("Invalid move")
        successor = Board()
        successor.grid_ = self.grid_.copy()
        i = getBottomIndex(self.grid_, move)
        # playable move
        if i >= 0:
            successor.grid_[i, move] = player
        # unplayable move
        else:
            raise("Unplayable move")
        return successor
    
    def isWin(self, player):
        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.int8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        for kernel in detection_kernels:
            if (convolve2d(self.grid_==player, kernel, mode="valid") == 4).any():
                return True
        return False


    def isLose(self, player):
        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.int8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        for kernel in detection_kernels:
            if (convolve2d(self.grid_==player * -1, kernel, mode="valid") == 4).any():
                return True
        return False

    def display(self):
        n, m = self.grid_.shape
        print("-"*36)
        for i in range(n):
            r = "|"
            for j in range(m):
                r += "{:^4}".format(self.grid_[i, j]) + "|"
            print(r)
        print("-"*36)




def getBottomIndex(board, move):
    i = -1
    while i < board.shape[0] - 1:
        if board[i+1, move] == 0:
            i += 1
        else:
            return i
    return i
    

if __name__ == "__main__":
    np.array([[  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0]])
    A = Board()
    A.grid_ = np.array([[  0,  0,  0,  1,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  1,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  1, -1, -1, -1, -1,  0,  0]])
    
    print(A.getMoves())