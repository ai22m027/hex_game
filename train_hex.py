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

TRAINING_ITERATION = 0 # Current training iteration
# NN_FN required if TRAINING_ITERATION > 0 and SELFPLAY = TRUE
NN_FN = None # 'data/model/Hex_Model_v0.h5'
# NEW_NN_FN required if TRAINING = FALSE and EVALUATION = TRUE
NEW_NN_FN = None # 'data/model/Hex_Model.h5' #'data/model/Hex_Model10_12-Feb-2021(14:50:36).h5'
SELFPLAY = False            # If True self-play phase will be executed
TRAINING = True            # If True training phase will be executed
EVALUATION = False          # If True evaluation phase will be executed

# Final evaluation of models after running several training iterations 
FINAL_EVALUATION = False         # If True final evaluation will be performed
fe_model_nums = list(range(10+1))  # Iteration number of models to evaluate


# %% Imports
import os
from training_pipeline import record_params
if SELFPLAY or EVALUATION or FINAL_EVALUATION: # Force Keras to use CPU
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
if EVALUATION:
    from training_pipeline import tournament_Hex
if FINAL_EVALUATION:
    from training_pipeline import final_evaluation
    
    
# %% SELF-PLAY STAGE
# Use random rollouts to generate first dataset
NEURAL_NET = False if TRAINING_ITERATION == 0 else True
          
selfplay_kwargs = {
'BOARD_SIZE' : BOARD_SIZE, # Size of the Hex Board
'TRAINING_ITERATION' : TRAINING_ITERATION,
'NN_FN' : NN_FN,
'NUM_SELFPLAY_GAMES' : 20,
'NUM_CPUS' : 1              # Number of CPUs to use for parallel self-play
}

mcts_kwargs = { # Parameters for MCTS used in generating data
'GAME_ENV' : None,          # Game environment loaded in function for multiprocessing
'UCT_C' : 4,                # Constant C used to calculate UCT value
'CONSTRAINT' : 'rollout',   # Constraint can be 'rollout' or 'time'
'BUDGET' : 2000,            # Maximum number of rollouts or time in seconds
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

if SELFPLAY:
    chk_data = generate_Hex_data(selfplay_kwargs, mcts_kwargs)
    data_fns = chk_data.generate_data()
    record_params('selfplay', **{**selfplay_kwargs, **mcts_kwargs})


# %% TRAINING STAGE
training_kwargs = {     # Parameters used to train neural network
'BOARD_SIZE' : BOARD_SIZE, # Size of the Hex Board
'TRAINING_ITERATION' : TRAINING_ITERATION,
'NN_BASE_LR' : 1e-5,    # Neural network minimum learning rate for CLR
'NN_MAX_LR' : 1e-3,     # Neural network maximum learning rate for CLR
'CLR_SS_COEFF' : 4,     # CLR step-size coefficient
'BATCH_SIZE' : 32,     # Batch size for training neural network
'EPOCHS' : 100,         # Maximum number of training epochs
'CONV_REG' : 0.001,     # L2 regularization term for Conv2D layers
'DENSE_REG' : 0.001,    # L2 regularization term for Dense layers
'NUM_KERNELS' : 128,    # Number of Conv2D kernels in "body" of NN
'VAL_SPLIT' : 0.1,     # Fraction of training data to use for validation
'MIN_DELTA' : 0.001, # Min amount val loss must decrease to prevent stopping
'PATIENCE' : 500,    # Number of epochs of stagnation before stopping training
'POLICY_LOSS_WEIGHT' : 1.0, # Weighting given to policy head loss
'VALUE_LOSS_WEIGHT' : 1.0,  # Weighting given to value head loss
'SLIDING_WINDOW' : 1 # Number self-play iterations to include in training data
}

FIND_LR = False # Set True to find learning rate range prior to training
SLIDING_WINDOW = training_kwargs['SLIDING_WINDOW'] 

if TRAINING:
    # Load training data

    training_data = []
    data_path = r"data/training_data/Hex_data_" + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + "_" + str(selfplay_kwargs['NUM_SELFPLAY_GAMES'] ) + "games" + ".pkl"
    
    training_data.extend(load_training_data(data_path))

    # Load NN model
    if TRAINING_ITERATION == 0: 
        nn = create_nn(**training_kwargs)
        NN_FN = save_nn_to_disk(nn, BOARD_SIZE)
    else:
        nn = load_model(NN_FN)
    
    # Determine LR range or begin training
    if FIND_LR:
        run_lr_finder(training_data, start_lr=1e-8, end_lr=0.1, 
                      num_epochs=100, **training_kwargs) # Find LR range for CLR   
    else:
        # Train NN
        history, NEW_NN_FN = train_nn(training_data, nn, **training_kwargs)
        plot_filename = plot_history(history, nn, TRAINING_ITERATION, BOARD_SIZE)
        training_kwargs['OLD_NN_FN'] = NN_FN
        training_kwargs['NEW_NN_FN'] = NEW_NN_FN
        record_params('training', **training_kwargs)

    
# %% EVALUATION STAGE
tourney_kwargs = {
'BOARD_SIZE' : BOARD_SIZE, # Size of the Hex Board
'TRAINING_ITERATION' : TRAINING_ITERATION,
'OLD_NN_FN' : NN_FN,
'NEW_NN_FN' : NEW_NN_FN,   
'TOURNEY_GAMES' : 2,        # Number of games in tournament between NNs
'NUM_CPUS' : 1              # Number of CPUs to use for parallel tourney play
}

tourney_mcts_kwargs = {     # Parameters for MCTS used in tournament
'NN_FN' : NEW_NN_FN,
'UCT_C' : 4,                # Constant C used to calculate UCT value
'CONSTRAINT' : 'rollout',   # Constraint can be 'rollout' or 'time'
'BUDGET' : 200,             # Maximum number of rollouts or time in seconds
'MULTIPROC' : False,        # Enable multiprocessing
'NEURAL_NET' : True,        # If False uses random rollouts instead of NN
'VERBOSE' : False,          # MCTS prints search start/stop messages if True
'TRAINING' : False,         # True if self-play, False if competitive play
'DIRICHLET_ALPHA' : 1.0,    # Used to add noise to prior probs of actions
'DIRICHLET_EPSILON' : 0.25, # Fraction of noise added to prior probs of actions  
'TEMPERATURE_TAU' : 0,      # Initial value of temperature Tau
'TEMPERATURE_DECAY' : 0,    # Linear decay of Tau per move
'TEMP_DECAY_DELAY' : 0      # Move count before beginning decay of Tau value
}

if EVALUATION:
    print('Beginning tournament between {} and {}!'.format(NEW_NN_FN, NN_FN))
    tourney = tournament_Hex(tourney_kwargs, tourney_mcts_kwargs)
    tourney_fn = tourney.start_tournament()
    record_params('evaluation', **{**tourney_kwargs, **tourney_mcts_kwargs})
    
    
# %% FINAL EVALUATION
if FINAL_EVALUATION:
    fe = final_evaluation(fe_model_nums, tourney_kwargs, tourney_mcts_kwargs)
    fe.start_evaluation(num_cpus=4)
    record_params('final', **{**tourney_kwargs, **tourney_mcts_kwargs})