from player import Player
import numpy as np
from MCTS import Node

class mctsPlayer(Player):
    '''
    Drop the disc randomly on any of the columns of the board
    '''
    
    def __init__(self, rows, cols, connect_number, 
                 timeout_setup, timeout_move, max_invalid_moves, 
                 cylinder):
        self.rows = rows
        self.cols = cols
        self.connect_number = connect_number
        self.timeout_setup = timeout_setup
        self.timeout_move = timeout_move
        self.max_invalid_moves = max_invalid_moves
        self.cylinder = cylinder

    def setup(self,piece_color):
        self.piece_color=piece_color

    def play(self, board: np.ndarray):
        """
        Given a 2D array representing the game board, return an integer value (0,1,2,...,number of columns-1) corresponding to
        the column of the board where you want to drop your disc.
        The coordinates of the board increase along the right and down directions. 

        Parameters
        ----------
        board : np.ndarray
            A 2D array where 0s represent empty slots, +1s represent your pieces,
            and -1s represent the opposing player's pieces.

                `index   0   1   2   . column` \\
                `--------------------------` \\
                `0   |   0.  0.  0.  top` \\
                `1   |   -1  0.  0.  .` \\
                `2   |   +1  -1  -1  .` \\
                `.   |   -1  +1  +1  .` \\
                `row |   left        bottom/right`

        Returns
        -------
        integer corresponding to the column of the board where you want to drop your disc.
        """
        root = Node(board, player=1) # 1 is ourselves

        return self.mcts(root)

    def mcts(self, root, iterations=50):
        for _ in range(iterations):
            node = root 
            while not node.check_results(node.get_board()) and node.untried_moves == []:
                node = node.select_child() 

            if node.untried_moves != []:
                node = node.expand()  # get to the child node for expansion

            result = node.rollout() # Expand and get result

            while node is not None:
                node.update(result)
                node = node.parent # Back propagate through the path

        return max(root.children, key=lambda c: c.wins / c.visits if c.visits else 0).move