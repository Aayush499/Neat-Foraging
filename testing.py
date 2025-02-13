from Forage import Game
import pygame
import neat
import os
import time
import pickle
import matplotlib.pyplot as plt
from main import test_best_network
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-reduced-input')
for i in range(5):
    test_best_network(config)
