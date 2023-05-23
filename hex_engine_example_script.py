#make sure that the module is located somewhere where your Python system looks for packages
import sys

#importing the module
from fhtw_hex import hex_engine as engine

#initializing a game object
game = engine.hexPosition()

#play the game against a random player, human plays 'black'
#game.human_vs_machine(human_player=-1, machine=None)

#this is how you will provide the agent you generate during the group project
from fhtw_hex import example as eg
from HexAgent import HexAgent

agent = HexAgent()

#play the game against the example agent, human play 'white'
game.human_vs_machine(human_player=1, machine=agent.predict)
