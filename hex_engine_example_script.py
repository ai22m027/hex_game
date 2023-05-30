#make sure that the module is located somewhere where your Python system looks for packages
import sys

#importing the module
from fhtw_hex import hex_engine as engine


BOARD_SIZE = 4
#initializing a game object
game = engine.hexPosition(size=BOARD_SIZE)

#play the game against a random player, human plays 'black'
#game.human_vs_machine(human_player=-1, machine=None)

#this is how you will provide the agent you generate during the group project
from fhtw_hex import example as eg
from HexAgent import HexAgent

agent = HexAgent(BOARD_SIZE=BOARD_SIZE)

#play the game against the example agent, human play 'white'
#game.human_vs_machine(machine1=agent.predict, machine2=None)

#game.human_vs_machine(human_player=-1, machine=agent.predict)
game.machine_vs_machine(machine1=None, machine2=agent.predict)
