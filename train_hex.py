#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# train_Hex.py
#
# Revision:     1.10
# Date:         1/26/2021
# Author:       Alex
#
# Purpose:      Trains a neural network to play Hex using self-play data.  
#               The initial dataset can be generated via randomized rollouts.
#               Once a neural network has been trained on the initial dataset,
#               subsequent datasets can be generated using the neural network's
#               self-play.
#
# Inputs:
# 1. The iteration of the training pipeline (start with TRAINING_ITERATION = 0)
# 2. The filename of the previous iteration's trained neural network (NN_FN)
# 3. Which phases of the training pipeline to run
# 4. Various parameters for each phase of the training pipeline
#
# Outputs:
# 1. Saves self-play data to /data/training_data
# 2. Saves the trained neural networks to /data/model.
# 3. Saves plots of the NN training loss to /data/plots.
# 4. Saves tournament results to /data/tournament_results
#
# Notes:
# 1. Self-play forces Keras to use CPU inferences, and the IPython kernel must
#    be restarted between the self-play and training phases to enable the GPU.
#    The same is true between the training and evaluation phases.
#
###############################################################################
"""

# %% Set training pipeline parameters

BOARD_SIZE = 7 # Size of the Hex Board
player = 0   # ONLY FOR TRAINING NECESSARY! Who should be trained? White Player: 0 | Black Player: 1

SELFPLAY = False            # If True self-play phase will be executed
TRAINING = True            # If True training phase will be executed


# %% Imports
import os
from training_pipeline import record_params
if SELFPLAY: # Force Keras to use CPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if SELFPLAY: 
    from training_pipeline import generate_Hex_data
if TRAINING:
    from training_pipeline import merge_data
    from training_pipeline import load_training_data
    from training_pipeline import create_nn
    from training_pipeline import save_nn_to_disk
    from training_pipeline import run_lr_finder
    from training_pipeline import train_nn
    from training_pipeline import plot_history
    from training_pipeline import create_timestamp
    from tensorflow.keras.models import load_model
    
# %% SELF-PLAY STAGE
selfplay_kwargs = {
'BOARD_SIZE' : BOARD_SIZE, # Size of the Hex Board
'NUM_SELFPLAY_GAMES' : 1000,
'NUM_CPUS' : 1              # Number of CPUs to use for parallel self-play
}

mcts_kwargs = { # Parameters for MCTS used in generating data
'GAME_ENV' : None,          # Game environment loaded in function for multiprocessing
'UCT_C' : 4,                # Constant C used to calculate UCT value
'CONSTRAINT' : 'rollout',   # Constraint can be 'rollout' or 'time'
'BUDGET' : 200,             # Maximum number of rollouts or time in seconds
'MULTIPROC' : False,        # Enable multiprocessing
'NEURAL_NET' : False,       # If False uses random rollouts instead of NN
'VERBOSE' : False,          # MCTS prints search start/stop messages if True
'TRAINING' : True,          # True if self-play, False if competitive play
'DIRICHLET_ALPHA' : 1.0,    # Used to add noise to prior probs of actions
'DIRICHLET_EPSILON' : 0.25, # Fraction of noise added to prior probs of actions   
'TEMPERATURE_TAU' : 1.0,    # Initial value of temperature Tau
'TEMPERATURE_DECAY' : 0.1,  # Linear decay of Tau per move
'TEMP_DECAY_DELAY' : 10     # Move count before beginning decay of Tau value
}

games = str(selfplay_kwargs['NUM_SELFPLAY_GAMES'] )
budget = str(mcts_kwargs['BUDGET'] ) 

if SELFPLAY:
    chk_data = generate_Hex_data(selfplay_kwargs, mcts_kwargs)
    data_fns = chk_data.generate_data()
    record_params('selfplay', games, budget, player, **{**selfplay_kwargs, **mcts_kwargs})


# %% TRAINING STAGE
training_kwargs = {     # Parameters used to train neural network
'BOARD_SIZE' : BOARD_SIZE, # Size of the Hex Board
'NN_BASE_LR' : 1e-4,    # Neural network minimum learning rate
'NN_MAX_LR' : 1e-5,     # Neural network maximum learning rate
'CLR_SS_COEFF' : 4,     # CLR step-size coefficient
'BATCH_SIZE' : 128,     # Batch size for training neural network
'EPOCHS' : 2000,         # Maximum number of training epochs
'CONV_NET' : True,      # Use Convolutional NN? Else: Fully Connected
'CONV_REG' : 0.001,     # L2 regularization term for Conv2D layers
'DENSE_REG' : 0.001,    # L2 regularization term for Dense layers
'NUM_KERNELS' : 256,    # Number of Conv2D kernels in "body" of NN
'VAL_SPLIT' : 0.1,     # Fraction of training data to use for validation
'MIN_DELTA' : 0.001, # Min amount val loss must decrease to prevent stopping
'PATIENCE' : 100,    # Number of epochs of stagnation before stopping training
}

FIND_LR = False # Set True to find learning rate range prior to training

if TRAINING:
    # Load training data
    training_data = []
    data_path = r"data/training_data/Hex_data_" + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + "_" + games + "games_" + budget + "budget" + ".pkl"
    training_data.extend(load_training_data(data_path))

    # Load NN model
    nn = create_nn(**training_kwargs)
    save_nn_to_disk(nn, BOARD_SIZE, games, budget, player)

    
    # Determine LR range or begin training
    if FIND_LR:
        run_lr_finder(training_data, start_lr=1e-8, end_lr=0.1, 
                      num_epochs=100, player=player, **training_kwargs) # Find LR range for CLR   
    else:
        # Train NN
        history = train_nn(training_data, nn, games, budget, player, **training_kwargs)
        plot_filename = plot_history(history, nn, BOARD_SIZE, games, budget, player)
        record_params('training', games, budget, player, **training_kwargs)

