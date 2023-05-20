#This script uses some packages from the python standard library:
#copy
#random
#pickle
#Of the above only 'copy' is necessary for basic functionality
from copy import deepcopy
import numpy as np

class Hex (object):
    """
    Objects of this class correspond to a game of Hex.
    
    Attributes
    ----------
    size : int 
        The size of the board. The board is 'size*size'.
    board : list[list[int]]
        An array representing the hex board. '0' means empty. '1' means 'white'. '-1' means 'black'.
    player : int
        The player who is currently required to make a moove. '1' means 'white'. '-1' means 'black'.
    winner : int
        Whether the game is won and by whom. '0' means 'no winner'. '1' means 'white' has won. '-1' means 'black' has won.
    history : list[list[list[int]]]
        A list of board-state arrays. Stores the history of play.    
    """
    def __init__ (self, size=7, neural_net=None):
        #enforce lower and upper bound on size
        self.size = max(2,min(size,26))
        #attributes encoding a game state
        # state 0: white player locations array
        # state 1: black player locations array
        # state 2: player id (0 or 1)
        # state 3: possible moves (free fields)
        # state 4: parent move leading to current node
        self.state = np.zeros((3 + 1 + 1, self.size, self.size), dtype=int)
        self.state[3] = 1 # All moves possible (empty board)
        self.history = [self.state]
        self.legal_next_states = self.get_legal_next_states(self.history)
        self.done = False
        self.outcome = None
        self.move_count = 0
        self.neural_net = neural_net

    def reset (self):
        """
        This method resets the hex board. All stones are removed from the board and the history is cleared.
        """
        self.state = np.zeros((3 + 1 + 1, self.size, self.size), dtype=int)
        self.state[3] = 1 # All moves possible (empty board)
        self.history = [self.state]
        self.legal_next_states = self.get_legal_next_states(self.history)
        self.done = False
        self.outcome = None
        self.move_count = 0

    def step (self, next_state):
        """
        This method enacts a moove.
        The variable 'coordinates' is a tuple of board coordinates.
        The variable 'player_num' is either 1 (white) or -1 (black).
        """
        if any((next_state == x).all() for x in self.legal_next_states):
            self.state = next_state
            self.history.append(self.state)
            self.done, self.outcome = self.determine_outcome(self.history, verbose=False)
            self.legal_next_states = self.get_legal_next_states(self.history)
            self.move_count += 1
            return self.state, self.outcome, self.done
        else:
            raise ValueError('These coordinates already contain a stone! (Invalid Move)')
    
    def get_legal_next_states (self, history):
        """If the game is not done, return a list of legal next moves given
        a board state as input.  The next moves are actually board states;
        the move to achieve those states is implied.
        """
        done, outcome = self.determine_outcome(history, verbose=False)
        if done == True: return [] # Game over
        state = history[-1]
        player = int(state[2,0,0])
        board = state[0] + state[1] # Combine player 1 and 2's pieces
        x_coords, y_coords = np.where(board == 0) # Find empty squares
        legal_next_states = []
        for x, y in zip(x_coords, y_coords):
            next_state = deepcopy(state)
            next_state[2] = 1 - player # Toggle player
            next_state[player, x, y] = 1 # Player chooses available empty square
            next_state[3, x, y] = 0 # Set field to zero, move here not possible anymore 
            next_state[4,0,0] = x # x move
            next_state[4,0,1] = y # y move
            legal_next_states.append(next_state)
        return legal_next_states
    
    def current_player(self, state):
        """Return which player's turn it is for a given input state."""
        player = int(state[2,0,0])
        if player == 0:
            return 'White Player'
        else:
            return 'Black Player'
        
    def predict(self, state):
        """Use the supplied neural network to predict the Q-value of a given
        state, as well as the prior probabilities of its child states.
        Masks any non-valid probabilities, and re-normalizes the remaining
        probabilities to sum to 1.
        """
        nn_inp = np.moveaxis(state[:4],0,-1)
        nn_inp = nn_inp.reshape(1,self.size,self.size,4)
        prob_planes, q_value = self.neural_net.predict(nn_inp)
        prob_planes, q_value = prob_planes[0].reshape((self.size,self.size,1)), q_value[0][0] 
        action_mask = state[3]
        prob_planes = np.squeeze(prob_planes) * action_mask
        prob_planes = prob_planes / np.sum(prob_planes)

        return prob_planes, q_value

    def set_prior_probs(self, child_nodes, prob_planes):
        """Takes as input a list of the parent node's child nodes, and the
        probability vector generated by running the parent's state through the
        neural network.  Assigns each child node its corresponding prior
        probability as predicted by the neural network.
        """
        for child in child_nodes:
            x = int(child.state[4,0,1])
            y = int(child.state[4,0,2])
            child._prior_prob = prob_planes[x, y]
            
    def _get_adjacent (self, coordinates):
        """
        Helper function to obtain adjacent cells in the board array.
        Used in position evaluation to construct paths through the board.
        """
        u = (coordinates[0]-1, coordinates[1])
        d = (coordinates[0]+1, coordinates[1])
        r = (coordinates[0], coordinates[1]-1)
        l = (coordinates[0], coordinates[1]+1)
        ur = (coordinates[0]-1, coordinates[1]+1)
        dl = (coordinates[0]+1, coordinates[1]-1)
        return [pair for pair in [u,d,r,l,ur,dl] if max(pair[0], pair[1]) <= self.size-1 and min(pair[0], pair[1]) >= 0]

    def _prolong_path (self, state, path):
        """
        A helper function used for board evaluation.
        """
        white_board = deepcopy(state[0])
        black_board = deepcopy(state[1])
        white_board[white_board==1] = 1 
        black_board[black_board==1] = -1 

        board  = white_board+black_board

        player = int(board[path[-1][0],path[-1][1]])
        candidates = self._get_adjacent(path[-1])
        #preclude loops
        candidates = [cand for cand in candidates if cand not in path]
        candidates = [cand for cand in candidates if board[cand[0],cand[1]] == player]
        return [path+[cand] for cand in candidates]
    
    def determine_outcome (self, history, verbose=False):
        """
        Evaluates the board position and adjusts the 'winner' attribute of the object accordingly.
        """
        state = history[-1]
        
        white_done = self._evaluate_white(state, verbose=verbose)
        black_done = self._evaluate_black(state, verbose=verbose)  
        
        if white_done:
            done = True
            outcome = 'White Player Wins!'
        elif black_done:
            done = True
            outcome = 'Black Player Wins!'
        else: 
            done = False # Game is still in progress
            outcome = None
            
        return done, outcome

    def _evaluate_white (self, state, verbose):
        """
        Evaluate whether the board position is a win for player '1'. Uses breadth first search.
        If verbose=True a winning path will be printed to the standard output (if one exists).
        This method may be time-consuming for huge board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if int(state[0,i,0]) == 1:
                paths.append([(i,0)])
                visited.append([(i,0)])
        while True:
            if len(paths) == 0:
                return False
            for path in paths:
                prolongations = self._prolong_path(state, path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][1] == self.size-1:
                        if verbose:
                            print("A winning path for 'white' ('1'):\n",new)
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])

    def _evaluate_black (self, state, verbose):
        """
        Evaluate whether the board position is a win for player '-1'. Uses breadth first search.
        If verbose=True a winning path will be printed to the standard output (if one exists).
        This method may be time-consuming for huge board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if int(state[1,0,i]) == 1:
                paths.append([(0,i)])
                visited.append([(0,i)])
        while True:
            if len(paths) == 0:
                return False
            for path in paths:
                prolongations = self._prolong_path(state, path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][0] == self.size-1:
                        if verbose:
                            print("A winning path for 'black' ('-1'):\n",new)
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])

           
    def print (self, invert_colors=True):
        """
        This method prints a visualization of the hex board to the standard output.
        If the standard output prints black text on a white background, one must set invert_colors=False.
        """
        white_board = deepcopy(self.state[0])
        black_board = deepcopy(self.state[1])
        white_board[white_board==1] = 1 
        black_board[black_board==1] = -1 
        
        board  = white_board+black_board

        names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        indent = 0
        headings = " "*5+(" "*3).join(names[:self.size])
        print(headings)
        tops = " "*5+(" "*3).join("_"*self.size)
        print(tops)
        roof = " "*4+"/ \\"+"_/ \\"*(self.size-1)
        print(roof)
        #color mapping inverted by default for display in terminal.
        if invert_colors:
            color_mapping = lambda i: " " if i==0 else ("\u25CB" if i== -1 else "\u25CF")
        else:
            color_mapping = lambda i: " " if i==0 else ("\u25CF" if i== -1 else "\u25CB")
        for r in range(self.size):
            row_mid = " "*indent
            row_mid += "   | "
            row_mid += " | ".join(map(color_mapping,board[r]))
            row_mid += " | {} ".format(r+1)
            print(row_mid)
            row_bottom = " "*indent
            row_bottom += " "*3+" \\_/"*self.size
            if r<self.size-1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " "*(indent-2)+headings
        print(headings)
                     
        
