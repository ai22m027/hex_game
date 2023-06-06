import numpy as np
from keras.models import load_model

class HexAgent():
    def __init__(self, BOARD_SIZE=7, games=100, budget=100, player=1 ) -> None:
        #nn_path: str = "data/model/Hex_Model_7x7.h5"
        self.BOARD_SIZE = BOARD_SIZE
        if player == 0:
            player = "white"
        elif player == 1:
            player = "black"
        else:
            raise ValueError("Player must be 0 (white) or 1 (black)!")
        
        self.nn = load_model(r"data/model/Hex_Model_" + str(BOARD_SIZE) + "x"+ str(BOARD_SIZE) + "_" + str(games) + "games_" + str(budget) + "budget_" + player + ".h5")
        pass
    
    def predict(self, state, possible_actions):
        if np.count_nonzero(state) > self.BOARD_SIZE*self.BOARD_SIZE / 3:
            #print("NN turn")
            nn_inp = np.squeeze(state)[np.newaxis, :, :, np.newaxis]
            prob_planes = self.nn.predict(nn_inp)
            prob_planes = prob_planes.reshape((1, self.BOARD_SIZE,self.BOARD_SIZE))
  
        else:
            #print("Random turn")
            prob_planes = np.random.rand(1,self.BOARD_SIZE,self.BOARD_SIZE)

        action_mask = (np.squeeze(state) == 0) * 1
        prob_planes = np.squeeze(prob_planes) * action_mask

        return np.unravel_index(prob_planes.argmax(), prob_planes.shape)