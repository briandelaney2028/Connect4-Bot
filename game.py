import sys
from Board import Board
from Agents import *
import random

if __name__ == "__main__":
    num_players = 1
    player = 1
    move = 0
    i = 1
    args = sys.argv
    while i < len(args):
        # number of players
        if args[i] == "-p":
            i += 1
            try:
                num_players = int(args[i])
                if num_players not in {0, 1, 2}:
                    raise("Not valid number of players -> {0, 1, 2}")
            except:
                raise("Number of players must be int")
        # decide who goes first
        if args[i] == "-f":
            i += 1
            try:
                player = int(args[i])
                if player not in {-1, 1}:
                    raise("Not valid player -> {-1, 1}")
            except:
                raise("Player must be int -> {-1, 1}")

        i += 1
    
    gameState = Board()     
    player1 = UserAgent(1)
    # player1 = MinimaxAgent(1, depth=2)
    # player2 = ReflexAgent(-1)
    # player2 = MinimaxAgent(-1, depth=1)
    player2 = AlphaBetaAgent(-1, depth=2)
    gameState.display()

    while move <= 42:
        if player == 1:
            action = player1.getAction(gameState)
        else:
            action = player2.getAction(gameState)
        gameState = gameState.getSuccessor(player, action)
        gameState.display()
        if gameState.isWin(player):
            print("Player {} wins!".format(player))
            break
        
        player *= -1
        move += 1
    
    if move > 42: print("Draw!")