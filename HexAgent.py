import numpy as np
from keras.models import load_model

class HexAgent():
    def __init__(self, BOARD_SIZE=7 ) -> None:
        #nn_path: str = "data/model/Hex_Model_7x7.h5"
        self.BOARD_SIZE = BOARD_SIZE
        self.nn = load_model(r"data/model/Hex_Model_" + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + ".h5")
        pass
    
    def __convert_state(self, state_in: list) -> np.array:
        """Convert the state provided by HexEngine to a usable state
        for our NN.

        Args:
            state (np.array): state provided by HexEngine

        Returns:
            np.array: usable state for our NN
        """
        board = np.array(state_in)

        state = np.zeros((5,self.BOARD_SIZE,self.BOARD_SIZE), int)
        state[0] = (board == 1) * 1    # white player
        state[1] = (board == -1) * 1   # black player
        state[2] = 0 #board[2] + 1  # player id 0 or 1
        state[3] = (board == 0) * 1    # possible moves

        return state
    
    def predict(self, state, possible_actions):
        conv_state = self.__convert_state(state)
        
        nn_inp = np.moveaxis(conv_state[:4],0,-1)
        nn_inp = nn_inp.reshape(1,self.BOARD_SIZE,self.BOARD_SIZE,4)
        
        prob_planes, q_value = self.nn.predict(nn_inp)
        # print(prob_planes.shape)
        # print(q_value.shape)
        
        prob_planes, q_value = prob_planes[0].reshape((self.BOARD_SIZE,self.BOARD_SIZE,1)), q_value[0][0] 
        action_mask = conv_state[3]
        prob_planes = np.squeeze(prob_planes) * action_mask
        prob_planes = prob_planes / np.sum(prob_planes)
        print(prob_planes)
 
        
        return np.unravel_index(prob_planes.argmax(), prob_planes.shape)