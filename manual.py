from Forage import Game
import pygame
import neat
import os
import time
import pickle
import matplotlib.pyplot as plt
from main import ForageTask

# 1. specify config file path
config_path = 'config-reduced-input'
# 2. define the path for the best network
winner_path = "best.pickle"

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)



def manual_testing(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Forage")
    foragetask = ForageTask(win, width, height)
    foragetask.manual_test(winner_net)
    
manual_testing(config)