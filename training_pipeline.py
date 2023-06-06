#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# training_pipeline.py
#
# Revision:     1.00
# Date:         11/27/2020
# Author:       Alex
#
# Purpose:      Contains classes to generate Hex training data and to 
#               create a tournament to compare the performance of two 
#               different trained neural networks.  
#
# Classes:
# 1. generate_Hex_data()   --Generates Hex training data through 
#                                 self-play. 
#
# 2. tournament_Hex()      --Pits two neural networks against each other 
#                                 in a Hex tournament and saves the 
#                                 result of the tournament to disk.
#
# Notes:
# 1. Training data saved to /data/training_data as a Pickle file.
# 2. Tournament results saved to /data/tournament_results.
# 3. Be sure to set the MCTS kwarg 'TRAINING' to False for competitive play.
#
###############################################################################
"""

from tensorflow.keras.utils import Sequence
from Hex import Hex
from CLR.clr_callback import CyclicLR
from LRFinder.keras_callback import LRFinder
import numpy as np
import pickle, os
from datetime import datetime
from tabulate import tabulate
import multiprocessing as mp
import matplotlib.pyplot as plt


# %% Functions
def create_nn(**kwargs):
    """Create a double-headed neural network used to learn to play Hex."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    creg = l2(kwargs['CONV_REG']) # Conv2D regularization param
    dreg = l2(kwargs['DENSE_REG']) # Dense regularization param
    num_kernels = kwargs['NUM_KERNELS'] # Num of Conv2D kernels in "body" of NN
    BOARD_SIZE = kwargs['BOARD_SIZE']
    conv_net = kwargs['CONV_NET']

    if conv_net:
        inputs = Input(shape = (BOARD_SIZE,BOARD_SIZE, 1))
        conv0 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                       use_bias = True, data_format='channels_last',
                       kernel_regularizer=creg, bias_regularizer=creg)(inputs)
        bn0 = BatchNormalization(axis=-1)(conv0)
        conv1 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                       use_bias = True, data_format='channels_last',
                       kernel_regularizer=creg, bias_regularizer=creg)(bn0)
        bn1 = BatchNormalization(axis=-1)(conv1)
        conv2 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                       use_bias = True, data_format='channels_last',
                       kernel_regularizer=creg, bias_regularizer=creg)(bn1)
        bn2 = BatchNormalization(axis=-1)(conv2)
        conv3 = Conv2D(num_kernels, (3, 3), padding='same', activation = 'relu', 
                       use_bias = True, data_format='channels_last',
                       kernel_regularizer=creg, bias_regularizer=creg)(bn2)
        bn3 = BatchNormalization(axis=-1)(conv3)
        
        # Create policy head
        policy_conv1 = Conv2D(num_kernels, (3,3), padding='same', activation = 'relu', 
                          use_bias = True, data_format='channels_last',
                          kernel_regularizer=creg, bias_regularizer=creg)(bn3)
        bn_pol1 = BatchNormalization(axis=-1)(policy_conv1)
        policy_conv2 = Conv2D(BOARD_SIZE*BOARD_SIZE, (1, 1), padding='same', activation = 'relu', 
                          use_bias = True, data_format='channels_last',
                          kernel_regularizer=creg, bias_regularizer=creg)(bn_pol1)
        bn_pol2 = BatchNormalization(axis=-1)(policy_conv2)
        policy_flat1 = Flatten()(bn_pol2)
        policy_output = Dense(BOARD_SIZE*BOARD_SIZE, activation = 'softmax', use_bias = True,
                      kernel_regularizer=dreg, bias_regularizer=dreg,
                      name='policy_head')(policy_flat1)
    else:
        inputs = Input(shape = (BOARD_SIZE, BOARD_SIZE, 1))
        
        _input = Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, 1))(inputs)
        
        l1 = Dense(200, activation = 'relu', use_bias = True)(_input)
        l1 = Dense(200, activation = 'relu', use_bias = True)(l1)
        
        policy_output = Dense(BOARD_SIZE*BOARD_SIZE, activation = 'softmax', use_bias = True,
                      name='policy_head')(l1)        
    
    # Compile model
    model = Model(inputs, policy_output)
    model.compile(loss={'policy_head' : 'categorical_crossentropy'}, optimizer=Adam())
    return model


def train_nn(training_data, neural_network, games, budget, train_player, **kwargs):
    """Trains neural network according to desired parameters."""
    import tensorflow as tf
    # Unpack kwargs
    PATIENCE = kwargs['PATIENCE']
    MIN_DELTA = kwargs['MIN_DELTA']
    VAL_SPLIT = kwargs['VAL_SPLIT']
    BOARD_SIZE = kwargs['BOARD_SIZE']
    BATCH_SIZE = kwargs['BATCH_SIZE']
    CLR_SS_COEFF = kwargs['CLR_SS_COEFF']
    NN_BASE_LR = kwargs['NN_BASE_LR']
    NN_MAX_LR = kwargs['NN_MAX_LR']
    EPOCHS = kwargs['EPOCHS']
    
    if train_player == 0:
        player = "white"
    elif train_player == 1:
        player = "black"
    else:
        raise ValueError("Player must be 0 (white) or 1 (black)!")
            
    # Create early stop callback for training
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              patience=PATIENCE, mode='min', 
                                              min_delta=MIN_DELTA, verbose=1)
    # Create model checkpoint callback to save best model
    filepath = r"data/model/Hex_Model_" + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + "_" + games + "games_" + budget + "budget_" + player + ".h5"
    save_best = tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                    monitor='val_loss', 
                                                    verbose=1, 
                                                    save_best_only=True,
                                                    save_weights_only=False, 
                                                    mode='auto', 
                                                    save_freq='epoch')
      
    # Create data generators for training
    np.random.shuffle(training_data) # Randomize order of training data
    if VAL_SPLIT > 0: # Split data into training and validation sets
        validation_data = training_data[-int(len(training_data)*VAL_SPLIT):]
        del training_data[-int(len(training_data)*VAL_SPLIT):]
        validation_generator = Keras_Generator(validation_data, BATCH_SIZE, train_player)
        validation_steps = len(validation_generator)
    else:
        validation_generator = None
        validation_steps = None
        

    # Create generator to feed training data to NN  
    training_generator = Keras_Generator(training_data, BATCH_SIZE, train_player)
    steps_per_epoch = len(training_generator)
    # Set CLR options
    clr_step_size = int(CLR_SS_COEFF * (len(training_data)/BATCH_SIZE))
    base_lr = NN_BASE_LR
    max_lr = NN_MAX_LR
    mode='triangular'
    # Define the CLR callback
    clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=clr_step_size, mode=mode)
    # Train NN using generators and callbacks
    history = neural_network.fit(x = training_generator,
                                  steps_per_epoch = steps_per_epoch,
                                  validation_data = validation_generator,
                                  validation_steps = validation_steps,
                                  epochs = EPOCHS,
                                  verbose = 1,
                                  shuffle = True,
                                  callbacks=[early_stop, clr, save_best])
    return history

def set_nn_lrate(neural_network, lrate):
    """Set the learning rate of an existing neural network."""
    from tensorflow.keras import backend as K
    K.set_value(neural_network.optimizer.learning_rate, lrate)
    
def save_nn_to_disk(neural_network, BOARD_SIZE, games, budget, player):
    """Save neural network to disk with timestamp and iteration in filename."""
        
    if player == 0:
        player = "white"
    elif player == 1:
        player = "black"
    else:
        raise ValueError("Player must be 0 (white) or 1 (black)!")
            
    filename = r"data/model/Hex_Model_" + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + "_" + games + "games_" + budget + "budget_" + player + ".h5"
    neural_network.save(filename)
    return filename
    
def create_timestamp():
    """Create timestamp string to be used in filenames."""
    timestamp = datetime.now(tz=None)
    timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
    return timestamp_str

def plot_history(history, nn, BOARD_SIZE, games, budget, player):
    """Plot of training loss versus training epoch and save to disk."""
    
    if player == 0:
        player = "white"
    elif player == 1:
        player = "black"
    else:
        raise ValueError("Player must be 0 (white) or 1 (black)!")
            
    legend = list(history.history.keys())
    for key in history.history.keys():
        plt.plot(history.history[key])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper right')
    plt.grid()
    filename = r"data/plots/Hex_Model_" + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + "_" + games + "games_" + budget + "budget_" + player + ".png"
    plt.draw()
    fig1 = plt.gcf()
    fig1.set_dpi(200)
    save_plots_dirname = "data/plots"
    if not os.path.exists(save_plots_dirname):
        os.makedirs(save_plots_dirname) 
    fig1.savefig(filename)
    plt.show()
    plt.close()
    return filename

def load_training_data(filename):
    """Load Pickle file and return training data."""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def record_params(phase, games, budget, player, **kwargs):
    """Document the parameters used in the training pipeline."""
    
    BOARD_SIZE = kwargs['BOARD_SIZE']
    
    if player == 0:
        player = "white"
    elif player == 1:
        player = "black"
    else:
        raise ValueError("Player must be 0 (white) or 1 (black)!")
    
    if phase == 'selfplay':
        save_training_dirname = "data/training_data"
        if not os.path.exists(save_training_dirname):
            os.makedirs(save_training_dirname)
        filename = 'data/training_data/Hex_SelfPlay_Params' + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + "_" + games + "games_" + budget + "budget_" + player + '.txt'
            
    elif phase == 'training':   
        save_model_dirname = "data/model"
        if not os.path.exists(save_model_dirname):
            os.makedirs(save_model_dirname) 
        filename = 'data/model/Hex_Training_Params' + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + "_" + games + "games_" + budget + "budget_" + player + '.txt'
            
    else:
        raise ValueError('Invalid phase!')
    # Write parameters to file
    with open(filename, 'w') as file:
        for key, val in kwargs.items():
            file.write('{} = {}\n'.format(key, val))
    
def run_lr_finder(training_data, start_lr, end_lr, num_epochs, player, **kwargs):
    """Linearly increase learning rate while training neural network over
    a number of epochs.  Outputs plot of training loss versus training
    iteration used to select the base and maximum learning rates used in CLR.
    """
    BATCH_SIZE = kwargs['BATCH_SIZE']
    np.random.shuffle(training_data) # Randomize order of training data
    model = create_nn(**kwargs)
    # Define LR finder callback
    lr_finder = LRFinder(min_lr=start_lr, max_lr=end_lr)
    # Create generator to feed training data to NN   
    training_generator = Keras_Generator(training_data, BATCH_SIZE, player)
    steps_per_epoch = len(training_generator)
    # Perform LR finder
    model.fit(x = training_generator,
                steps_per_epoch = steps_per_epoch,
                validation_data = None,
                validation_steps = None,
                epochs = num_epochs,
                verbose = 1,
                shuffle = True,
                callbacks=[lr_finder])

def save_merged_files(memory, iteration, timestamp):
    """Save training data to disk as a Pickle file."""
    filename = r'data/training_data/Hex_Data.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(memory, file)
    return filename

def merge_data(data_fns, iteration):
    """Merge multiple training files into a single file."""
    training_data = []        
    if isinstance(data_fns, list):
        for idx, fn in enumerate(data_fns):
            training_data.extend(load_training_data(fn))
       
    else:
        training_data.extend(load_training_data(data_fns))
    
    filename = save_merged_files(training_data, iteration, create_timestamp()) 
        
    return training_data


# %% Classes
class Keras_Generator(Sequence):
    """Generator to feed training/validation data to Keras fit() function."""
    def __init__(self, data, batch_size, train_player) :
        self.data = data
        self.batch_size = batch_size
        self.train_player = train_player
    
    def __len__(self):
        return (np.ceil(len(self.data) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx):
        """Splits data into features and labels.  Returns average of Q-values
        and Z-values for the value head labels.
        """
        data = self.data[idx * self.batch_size : (idx+1) * self.batch_size]
                
        # get data for Black Player Turn only and exclude (and not) end game state, and only take samples when minimum 4 stones are on the board
        states = np.array([e[0][0] - e[0][1] for e in data if e[0][2][0][0] == self.train_player and not e[1].all == 0 and np.count_nonzero(e[1]) > (e[0][0].shape[0]*e[0][0].shape[1] / 3)])  
        probs = np.array([np.array(e[1]).flatten() for e in data if e[0][2][0][0] == self.train_player and not e[1].all == 0 and np.count_nonzero(e[1]) > (e[0][0].shape[0]*e[0][0].shape[1] / 3)])
        
        #print(states, probs)
        
        return (states, probs)


class generate_Hex_data():
    """Class to generate Hex training data through self-play."""
    def __init__(self, selfplay_kwargs, mcts_kwargs):
        """Set trainng parameters and initialize MCTS class."""
        self.BOARD_SIZE = selfplay_kwargs['BOARD_SIZE']
        self.NUM_SELFPLAY_GAMES = selfplay_kwargs['NUM_SELFPLAY_GAMES']
        self.BUDGET = mcts_kwargs['BUDGET']
        self.num_cpus = selfplay_kwargs['NUM_CPUS']
        self.mcts_kwargs = mcts_kwargs
        MAX_PROCESSORS = mp.cpu_count()
        if self.num_cpus > MAX_PROCESSORS: self.num_cpus = MAX_PROCESSORS

    def generate_data(self):
        """Uses the multiprocessing module to parallelize self-play."""
        if self.num_cpus > 1:
            pool = mp.Pool(self.num_cpus)
            filenames = pool.map(self._generate_data, range(self.num_cpus))
            pool.close()
            pool.join()
        else:
            filenames = self._generate_data()
        return filenames  
        
    def _generate_data(self, process_num=0):
        """Generate Hex training data for a neural network through 
        self-play.  Plays the user-specified number of games, and returns the 
        data as a list of lists.  Each sub-list contains a game state, a
        probability planes, and the terminal reward of the episode from the 
        perspective of the state's current player.
        """
        np.random.seed()
        from tensorflow.keras.models import load_model
        from MCTS import MCTS
        from MCTS import MCTS_Node

        game_env = Hex(size=self.BOARD_SIZE, neural_net=None)
        
        self.mcts_kwargs['GAME_ENV'] = game_env
        MCTS(**self.mcts_kwargs) # Set MCTS parameters
        memory = []
        for _ in range(self.NUM_SELFPLAY_GAMES):
            print('Beginning game {} of {}!'.format(_+1, self.NUM_SELFPLAY_GAMES))
            experiences = []
            initial_state = game_env.state
            root_node1 = MCTS_Node(initial_state, parent=None)
            terminated_game = False
            parent_player = 'Black Player'
            while not game_env.done: # Game loop
                if game_env.current_player(game_env.state) == 'White Player':
                    if game_env.move_count != 0:  # Update P1 root node w/ P2's move
                        parent_player = MCTS.current_player(game_env.history[-2])
                        root_node1 = MCTS.new_root_node(best_child1)
                
                    MCTS.begin_tree_search(root_node1)
                    best_child1 = MCTS.best_child(root_node1)
                    game_env.step(best_child1.state)
                    prob_planes = self._create_prob_planes(root_node1)
                    if parent_player != root_node1.player:
                        qval = -root_node1.q  
                    else:
                        qval = root_node1.q
                        
                    experiences.append([root_node1.state, prob_planes, qval])
                    # print("White Player's Turn:")
                    # print(root_node1.state)
                    # print(prob_planes)
                    # game_env.print()
                    
                else:
                    if game_env.move_count == 1: # Initialize second player's MCTS node 
                       root_node2 = MCTS_Node(game_env.state, parent=None, 
                                              initial_state=initial_state)
                       parent_player = 'Black Player'
                    else: # Update P2 root node with P1's move
                        parent_player = MCTS.current_player(game_env.history[-2])
                        root_node2 = MCTS.new_root_node(best_child2)
                    MCTS.begin_tree_search(root_node2)
                    best_child2 = MCTS.best_child(root_node2)
                    game_env.step(best_child2.state)
                    prob_planes = self._create_prob_planes(root_node2)
                    if parent_player != root_node2.player:
                        qval = -root_node2.q  
                    else:
                        qval = root_node2.q
                    
                    experiences.append([root_node2.state, prob_planes, qval])
                    # print("Black Player's Turn:")
                    # print(game_p_state)
                    # print(prob_planes)
                    # game_env.print()
                    
            if not terminated_game: # Include terminal state
                prob_planes = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
                node_q = -1
                experiences.append([game_env.state, prob_planes, node_q])
            experiences = self._add_rewards(experiences, game_env.outcome)
            memory.extend(experiences)
            print('{} after {} moves!'.format(game_env.outcome, game_env.move_count))
            game_env.print()
            game_env.reset()
        if MCTS.multiproc: 
                MCTS.pool.close()
                MCTS.pool.join()
        filename = self._save_memory(memory, self.NUM_SELFPLAY_GAMES, self.BUDGET, 
                          self._create_timestamp(), process_num)
        
        return filename

    def _create_prob_planes(self, node):
        """Populate the probability planes used to train the neural network's
        policy head.  Uses the probabilities generated by the MCTS for each 
        child node of the given node.
        """
        prob_planes = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        for child in node.children:
            x = int(child.state[4,0,0])
            y = int(child.state[4,0,1])
            prob_planes[x, y] = child.n
        prob_planes /= np.sum(prob_planes)

        if not np.isclose(np.sum(prob_planes), 1): 
            raise ValueError('Probabilities do not sum to 1!')
        return prob_planes

    def _add_rewards(self, experiences, outcome):
        """Include a reward with every state based on the outcome of the 
        episode.  This is used to train the value head of the neural network by 
        providing the actual outcome of the game as training data.  Note that
        the rewards are not reversed like in the MCTS.
        """
        for experience in experiences:
            state = experience[0]
            player = int(state[2,0,0])
            if outcome == 'White Player Wins!':
                reward = 1 if player == 0 else -1
            elif outcome == 'Black Player Wins!':
                reward = 1 if player == 1 else -1
            experience.append(reward)
        return experiences

    def _save_memory(self, memory, games, budget, timestamp, process_num):
        """Save training data to disk as a Pickle file."""
    
        save_training_dirname = "data/training_data"
        
        if not os.path.exists(save_training_dirname):
            os.makedirs(save_training_dirname)
        
        filename = r"data/training_data/Hex_Data_" + str(self.BOARD_SIZE) + "x"+ str(self.BOARD_SIZE) + "_" + str(self.NUM_SELFPLAY_GAMES) + "games_" + str(self.BUDGET) + "budget" +  ".pkl"
        
        with open(filename, 'wb') as file:
            pickle.dump(memory, file)
            
        return filename
        
    def _create_timestamp(self):
        """Create timestamp string to be used in filenames."""
        timestamp = datetime.now(tz=None)
        timestamp_str = timestamp.strftime("%d-%b-%Y(%H:%M:%S)")
        return timestamp_str
    