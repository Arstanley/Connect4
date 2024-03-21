import numpy as np
import random
import math
from connect4 import Connect4Board

class Node:
    def __init__(self, board, player, move=None, parent=None):
        self.board = np.copy(board)  
        self.player = player  
        self.move = move  
        self.parent = parent  
        self.children = []  
        self.wins = 0  
        self.visits = 0  
        self.untried_moves = self.available_moves()  

        # Helper for checking results
        self.connect5 = Connect4Board(self.board.shape[0], self.board.shape[1], 5, True)
    
    def get_board(self):
        return self.board
    
    def available_moves(self):
        # Assuming the board's 0s represent available spots to play
        return [col for col in range(self.board.shape[1]) if 0 in self.board[:, col]]

    def update(self, result):
        self.visits += 1
        self.wins += result

    def select_child(self):
        # Select a child node using UCB
        ucb = [(child.wins / child.visits) + math.sqrt(2 * math.log(self.visits) / child.visits) for child in self.children]
        return self.children[np.argmax(ucb)]

    def check_results(self, board):
        # Check if the game is over (win/loss/draw)
        return 1 if (self.connect5.check_if_winner(board) is not None) else -1 if (self.connect5.check_if_winner(board * -1) is not None) else None

    def expand(self):
        move = self.untried_moves.pop()
        current_board = np.copy(self.board)
        next_board = self.simulate_move(current_board, move, self.player)
        child_node = Node(next_board, -self.player, move, self)
        self.children.append(child_node)
        return child_node

    def simulate_move(self, board, move, player):
        if player:
            if isinstance(move, (int,np.int32,np.int64)) and 0 <= move < self.board.shape[1]: # Copied from the connect4 class
                n_spots=sum(board[:, move]==0)
                if n_spots:
                    board[n_spots-1, move]=1
            return board  
        else:
            if isinstance(move, (int,np.int32,np.int64)) and 0 <= move < self.board.shape[1]:
                n_spots=sum(board[:,move]==0)
                if n_spots:
                    board[n_spots-1,move]=-1
            return board

    def rollout(self):
        # Simulate a random playthrough to a terminal state
        current_board = np.copy(self.board)
        current_player = self.player
        while True:
            available_moves = [col for col in range(current_board.shape[1]) if 0 in current_board[:, col]]
            if not available_moves:  # If no moves left, it's a draw
                return 0
            move = random.choice(available_moves)
            current_board = self.simulate_move(current_board, move, current_player)
            game_result = self.check_results(current_board)
            if game_result is not None:
                return game_result 
            current_player = -current_player