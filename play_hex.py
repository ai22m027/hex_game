#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# play_TTT.py
#
# Revision:     1.00
# Date:         11/07/2020
# Author:       Alex
#
# Purpose:      Plays a demonstration game of Tic-Tac-Toe using the Monte Carlo
#               Tree Search algorithm. 
#
# Inputs:
# 1. MCTS parameters, e.g. the computational constraints and UCT constant.
# 2. Player selection - human vs. MCTS algorithm or algorithm vs. algorithm.
#
# Outputs:
# 1. Text representations of the Tic-Tac-Toe game board and the MCTS tree.
# 2. An optional print out of the MCTS tree after each player's move.
#
# Notes:
# 1. Run this module to see a demonstration game of Tic-Tac-Toe played using 
#    the MCTS algorithm.
#
###############################################################################
"""
# %% Imports
from Hex import Hex
from MCTS import MCTS
from MCTS import MCTS_Node
from keras.models import load_model


BOARD_SIZE = 7 # Size of the Hex Board

# Set MCTS parameters
mcts_kwargs = {     # Parameters for MCTS used in tournament
'BOARD_SIZE' : BOARD_SIZE,
'NN_FN' : r"data/model/Hex_Model_" + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + ".h5",
'UCT_C' : 4,                # Constant C used to calculate UCT value
'CONSTRAINT' : 'rollout',   # Constraint can be 'rollout' or 'time'
'BUDGET' : 100,            # Maximum number of rollouts or time in seconds
'MULTIPROC' : False,        # Enable multiprocessing
'NEURAL_NET' : True,       # If False uses random rollouts instead of NN
'VERBOSE' : False,           # MCTS prints search start/stop messages if True
'TRAINING' : False,         # True if self-play, False if competitive play
'DIRICHLET_ALPHA' : 1.0,    # Used to add noise to prior probs of actions
'DIRICHLET_EPSILON' : 0.25, # Fraction of noise added to prior probs of actions  
'TEMPERATURE_TAU' : 0,      # Initial value of temperature Tau
'TEMPERATURE_DECAY' : 0,    # Linear decay of Tau per move
'TEMP_DECAY_DELAY' : 0      # Move count before beginning decay of Tau value
}

# %% Initialize game environment and MCTS class

if mcts_kwargs['NEURAL_NET']:
    nn = load_model(mcts_kwargs['NN_FN'])
    game_env = Hex(size=mcts_kwargs['BOARD_SIZE'], neural_net=nn)
else:
    game_env = Hex(size=mcts_kwargs['BOARD_SIZE'], neural_net=None)

# %% Functions
def get_human_input():
    """Print a list of legal next states for the human player, and return
    the player's selection.
    """
    legal_next_states = game_env.legal_next_states
    for idx, state in enumerate(legal_next_states):
        print(state[human_player_idx], '\t', idx, '\n')
    move_idx = int(input('Enter move index: '))
    game_env.step(legal_next_states[move_idx])
    return legal_next_states[move_idx]

mcts_kwargs['GAME_ENV'] = game_env
MCTS(**mcts_kwargs)    
initial_state = game_env.state
game_env.print()


# Choose whether to play against the MCTS or to pit them against each other
human_player1 = False # Set true to play against the MCTS algorithm as player 1
human_player2 = False # Or choose player 2
if human_player1 and human_player2: human_player2 = False
human_player_idx = 0 if human_player1 else 1
if not human_player1:
    root_node1 = MCTS_Node(initial_state, parent=None)    

print_trees = True # Choose whether to print root node's tree after every move
tree_depth = 1 # Number of layers of tree to print (warning: expands quickly!)


# %% Game loop
while not game_env.done:
    if game_env.current_player(game_env.state) == 'White Player':
        if human_player1: 
            human_move = get_human_input()
        else: # MCTS plays as player 1
            if game_env.move_count != 0:  # Update P1 root node w/ P2's move
                root_node1 = MCTS.new_root_node(best_child1)
            #print("root_node1", root_node1.state)
            MCTS.begin_tree_search(root_node1)
            best_child1 = MCTS.best_child(root_node1)
            #print("best_child1", best_child1.state)
            game_env.step(best_child1.state)
            if print_trees: MCTS.print_tree(root_node1,tree_depth)
            
    else:
        if human_player2: 
            human_move = get_human_input()
        else: # MCTS plays as player 2
            if game_env.move_count == 1: # Initialize second player's MCTS node 
               root_node2 = MCTS_Node(game_env.state, parent=None, 
                                      initial_state=initial_state)
            else: # Update P2 root node with P1's move
                root_node2 = MCTS.new_root_node(best_child2)
            MCTS.begin_tree_search(root_node2)
            #print("root_node2", root_node2.state)
            best_child2 = MCTS.best_child(root_node2)
            #print("best_child2", best_child2.state)
            game_env.step(best_child2.state)
            if print_trees: MCTS.print_tree(root_node2,tree_depth)

    game_env.print()        
    game_env._evaluate_white(game_env.state, verbose=True)
    game_env._evaluate_black(game_env.state, verbose=True)  