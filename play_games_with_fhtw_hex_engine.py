#make sure that the module is located somewhere where your Python system looks for packages
import sys

#importing the module
from fhtw_hex import hex_engine as engine

# Size of Board
BOARD_SIZE = 7
# Machine Player
machine_player = -1   # White Player = Machine: 1 | Black Player = Machine: -1 | VS each other: 0
# Training Games
games = 1000
# Training Budget
budget = 200
#initializing a game object
game = engine.hexPosition(size=BOARD_SIZE)

#play the game against a random player, human plays 'black'
#game.human_vs_machine(human_player=-1, machine=None)

#this is how you will provide the agent you generate during the group project
from fhtw_hex import example as eg
from HexAgent import HexAgent

if machine_player == 1:
    white_agent = HexAgent(BOARD_SIZE=BOARD_SIZE, games=games, budget=budget, player=0)
elif machine_player == -1:
    black_agent = HexAgent(BOARD_SIZE=BOARD_SIZE, games=games, budget=budget, player=1)
elif machine_player == 0:
    white_agent = HexAgent(BOARD_SIZE=BOARD_SIZE, games=games, budget=budget, player=0)
    black_agent = HexAgent(BOARD_SIZE=BOARD_SIZE, games=games, budget=budget, player=1)

#play the game against the example agent, human play 'white'
#game.human_vs_machine(machine1=agent.predict, machine2=None)

#game.human_vs_machine(human_player=1, machine=black_agent.predict)

white_wins = 0
black_wins = 0
n_games = 100

for i_game in range(n_games):  
    if machine_player == 1:
        winner = game.machine_vs_machine(machine1=white_agent.predict, machine2=None)
    elif machine_player == -1:
        winner = game.machine_vs_machine(machine1=None, machine2=black_agent.predict)
    elif machine_player == 0:
        winner = game.machine_vs_machine(machine1=white_agent.predict, machine2=black_agent.predict)
        
    if winner == 1:
        white_wins+=1
    elif winner == -1:
        black_wins+=1
    
    print("Game Number ", i_game+1, " finished!")

    print("White won ", str(white_wins), " times!")
    print("Black won ", str(black_wins), " times!")
    
print("White won ratio:", str(white_wins / (white_wins + black_wins)*100), "%")
print("Black won ratio:", str(black_wins / (white_wins + black_wins)*100), "%")
