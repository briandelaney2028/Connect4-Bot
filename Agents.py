import numpy as np
from scipy.signal import convolve2d
from Board import *

class Agent(object):
    def __init__(self, index=0, depth=3):
        self.index = index
        self.name = "Agent"
        self.depth = depth

    def getAction(self, state):
        raise("Not defined")


def EvaluationFunction(gameState, player, move):
    score = -99999
    # if win pick win
    if gameState.isWin(player):
        return float('inf')
    if gameState.isLose(player):
        return -float('inf')

    # get i value of move
    i = getBottomIndex(gameState.grid_, move) + 1

    # search horizontal
    if move == 3:
        horizontal = gameState.grid_[i, :]
    elif move < 3:
        horizontal = gameState.grid_[i, :move + 4]
    else:
        horizontal = gameState.grid_[i, move-3:]
    
    # search vertical
    if i <= 2:
        vertical = gameState.grid_[:i + 4, move]
    else:
        vertical = gameState.grid_[i-3:, move]
    
    # search / diagonal
    idx = i
    jdx = move
    diag1 = []
    while idx < 6 and abs(i-idx) < 4 and jdx >= 0:
        diag1.insert(0, gameState.grid_[idx, jdx])
        idx += 1
        jdx -= 1
    idx = i - 1
    jdx = move + 1
    while idx >= 0 and abs(i-1-idx) < 3 and jdx < 7:
        diag1.append(gameState.grid_[idx, jdx])
        idx -= 1
        jdx += 1
    
    # search \ diagonal
    idx = i
    jdx = move
    diag2 = []
    while idx >= 0 and abs(i-idx) < 4 and jdx >= 0:
        diag2.insert(0, gameState.grid_[idx, jdx])
        idx -= 1
        jdx -= 1
    idx = i + 1
    jdx = move + 1
    while idx < 6 and abs(i+1-idx) < 3 and jdx < 7:
        diag2.append(gameState.grid_[idx, jdx])
        idx += 1
        jdx += 1

    # search neighbors for patterns
    neighbors = [horizontal, vertical, diag1, diag2]
    
    block = 0
    threeOfaKind = 0
    twoOfaKind = 0
    threeBlock = 0
    for direction in neighbors: # check each of 4 directions
        # print(direction)
        # check for block
        idx = 0
        detection_kernels = []
        detection_kernels.append([1, -1, -1, -1])
        detection_kernels.append([-1, 1, -1, -1])
        detection_kernels.append([-1, -1, 1, -1])
        detection_kernels.append([-1, -1, -1, 1])
        while idx < len(direction) - len(detection_kernels[0]) + 1: # check each slot in direction
            for kernel in detection_kernels: # check each kernel on each slot
                if all([k * player == d for k, d in zip(kernel, direction[idx:idx+4])]):
                    block += 1
            idx += 1

        # check for 3 of a kind
        idx = 0
        detection_kernels = []
        detection_kernels.append([0, 1, 1, 1])
        detection_kernels.append([1, 0, 1, 1])
        detection_kernels.append([1, 1, 0, 1])
        detection_kernels.append([1, 1, 1, 0])
        while idx < len(direction) - len(detection_kernels[0]) + 1: # check each slot in direction
            for kernel in detection_kernels: # check each kernel on each slot
                # print(kernel, direction[idx:idx+4])
                # print([k == d for k, d in zip(kernel, direction[idx:idx+4])])
                if all([k * player == d for k, d in zip(kernel, direction[idx:idx+4])]):
                    threeOfaKind += 1
            idx += 1
        
        # check for two of a kind
        idx = 0
        detection_kernels = []
        detection_kernels.append([1, 1])
        detection_kernels.append([1, 0, 1])
        detection_kernels.append([0, 1, 1])
        while idx < len(direction) - len(detection_kernels[1]) + 1:
            for kernel in detection_kernels:
                if len(kernel) == 3:
                    if all([k * player == d for k, d in zip(kernel, direction[idx:idx+3])]):
                        twoOfaKind += 1
                else:
                    if all([k * player == d for k, d in zip(kernel, direction[idx:idx+2])]):
                        twoOfaKind += 1
            idx += 1
        
        # check for blocking 3
        idx = 0
        detection_kernels = []
        detection_kernels.append([1, -1, -1])
        detection_kernels.append([-1, 1, -1])
        detection_kernels.append([-1, -1, 1])
        while idx < len(direction) - len(detection_kernels[0]) + 1:
            for kernel in detection_kernels:
                if all([k * player == d for k, d in zip(kernel, direction[idx:idx+3])]):
                    threeBlock += 1
            idx += 1
      
    # print("Block:", block)
    # print("ThreeOfaKind:", threeOfaKind)
    # print("TwoOfaKind:", twoOfaKind)
    # print("ThreeBlock:", threeBlock)

    features = [block, threeOfaKind, twoOfaKind, threeBlock]
    weights = [99999, 100, 10, 1]

    score += sum([f * w for f, w in zip(features, weights)])

    # print("Score:", sum([f * w for f, w in zip(features, weights)]))

    return sum([f * w for f, w in zip(features, weights)])


class UserAgent(Agent):
    
    def getAction(self, state):
        s = "Player {} Turn! Enter column number: ".format(self.index)
        return int(input(s)) - 1

class ReflexAgent(Agent):

    def getAction(self, state):
        moves = state.getMoves()
        futureStates = [(state.getSuccessor(self.index, move), move) for move in moves]
        scores = [(EvaluationFunction(s[0], self.index, s[1]), s[1]) for s in futureStates]
        score, nextMove = max(scores)
        print("\nScore:", score)
        print("Action:", nextMove+1)
        if score == 0:
            return 3
        return nextMove

class MinimaxAgent(Agent):
    
    def getAction(self, state):
            # print("Depth:", self.depth)
            MM = self.Minimax(state, self.index, self.depth)
            # print("100% finished")
            print("Move:", MM[1] + 1)
            if MM[0] == 0:
                return 3
            return MM[1]
    
    def Minimax(self, gameState, player, depth, move=3):
        if player != self.index:
            depth -= 1
        # if depthlimit or game end
        if depth == -1 or gameState.isWin(player) or gameState.isLose(player):
            Eval = EvaluationFunction(gameState, player *-1, move) * player
            # print("Evaluation:", Eval, "Player:", player *-1, "Move:", move)
            return Eval, move
        
        # if max node
        elif player == self.index:
            # print("Player:", player)
            MM_value = -float('inf')
            MM_move = 4
            # moves
            moves = gameState.getMoves()
            percent = 0
            for futureMove in moves:
                # print("future move:",futureMove, "player:", player)
                if depth == self.depth-1:
                    percent += 1
                successor = gameState.getSuccessor(player, futureMove)
                if successor.isWin(player):
                    return -float('inf') * player, futureMove
                successorMM = self.Minimax(successor, player * -1, depth, futureMove)
                if successorMM[0] >= MM_value:
                    MM_value = successorMM[0]
                    MM_move = futureMove
            # print("MM:",MM_value, MM_move)
            return (MM_value, MM_move)
        
        # if min node
        else:
            # print("Player:", player)
            MM_value = float('inf')
            MM_move = 4
            # moves
            moves = gameState.getMoves()

            for futureMove in moves:
                # print("future move:",futureMove, "player:", player)
                successor = gameState.getSuccessor(player, futureMove)
                if successor.isWin(player):
                    return -float('inf') * player, futureMove
                successorMM = self.Minimax(successor, player * -1, depth, futureMove)
                # print(successorMM)
                if successorMM[0] <= MM_value:
                    MM_value = successorMM[0]
                    MM_move = futureMove
            # print("MM:",MM_value, MM_move)
            return (MM_value, MM_move)

class AlphaBetaAgent(Agent):
    
    def getAction(self, state):
            # print("Depth:", self.depth)
            AB = self.alphaBeta(state, self.index, self.depth, -9999999, 9999999)
            # print("100% finished")
            print("Move:", AB[1] + 1)
            if AB[0] == 0:
                return 3
            return AB[1]
    
    def alphaBeta(self, gameState, player, depth, alpha, beta, move=3):
        if player != self.index:
            depth -= 1
        # if depthlimit or game end
        if depth == -1 or gameState.isWin(player) or gameState.isLose(player):
            Eval = EvaluationFunction(gameState, player *-1, move) * player
            # print("Evaluation:", Eval, "Player:", player *-1, "Move:", move)
            return Eval, move
        
        # if max node
        elif player == self.index:
            # print("Player:", player)
            MM_value = -float('inf')
            MM_move = 4
            # moves
            moves = gameState.getMoves()
            percent = 0
            for futureMove in moves:
                # print("future move:",futureMove, "player:", player)
                if depth == self.depth-1:
                    percent += 1
                successor = gameState.getSuccessor(player, futureMove)
                if successor.isWin(player):
                    return -float('inf') * player, futureMove
                successorMM = self.alphaBeta(successor, player * -1, depth, alpha, beta, futureMove)
                if successorMM[0] >= MM_value:
                    MM_value = successorMM[0]
                    MM_move = futureMove
                alpha = max(alpha, MM_value)
                if MM_value > beta:
                    return MM_value, MM_move
                
            # print("MM:",MM_value, MM_move)
            return (MM_value, MM_move)
        
        # if min node
        else:
            # print("Player:", player)
            MM_value = float('inf')
            MM_move = 4
            # moves
            moves = gameState.getMoves()

            for futureMove in moves:
                # print("future move:",futureMove, "player:", player)
                successor = gameState.getSuccessor(player, futureMove)
                if successor.isWin(player):
                    return -float('inf') * player, futureMove
                successorMM = self.alphaBeta(successor, player * -1, depth, alpha, beta, futureMove)
                # print(successorMM)
                if successorMM[0] <= MM_value:
                    MM_value = successorMM[0]
                    MM_move = futureMove
                beta = min(beta, MM_value)
                if MM_value < alpha:
                    return MM_value, MM_move
            # print("MM:",MM_value, MM_move)
            return (MM_value, MM_move)

class ExpectimaxAgent(Agent):
    pass

def grid2col(grid, matchSize):
    n, m = grid.shape
    col = m - matchSize[1] + 1
    row = n - matchSize[0] + 1

    start = np.arange(matchSize[0])[:, None]*m + np.arange(matchSize[1])
    offset = np.arange(row)[:, None]*m + np.arange(col)

    return np.take(grid, start.ravel()[:, None] + offset.ravel())

if __name__ == "__main__":
    np.array([[  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0]])
    A = Board()
    A.grid_ = np.array([[  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0,  0,  0],
                        [  0, -1,  1,  1,  0,  0, -1]])
    print(A.grid_, end="\n\n")
    
    print(EvaluationFunction(A, -1, 3))