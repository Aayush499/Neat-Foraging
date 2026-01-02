from Forage import Game
import pygame
import neat
import os
import time
import pickle
import matplotlib.pyplot as plt
from main_parallel import ForageTask
from main_parallel import parser
import argparse

# 1. specify config file path
config_path = 'config-replication'
# 2. define the path for the best network
# winner_path = "checkpoints/biased_west/49"
winner_path = "best.pickle"

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)



def manual_testing(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)
    # p = neat.Checkpointer.restore_checkpoint(winner_path)

    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    width, height = 700, 700
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Forage")
    foragetask = ForageTask(win, width, height, arrangement_idx=0, )
    foragetask.manual_test(winner_net, auto=True)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

#run the argparser
if __name__ == "__main__":
    #run the parser function form main_parallel to get args
    # config_path = os.path.join(local_dir, 'config-replication-plateau')
    args = parser()
    global obstacles, particles, generations, movement_type, network_type, sub, best_file, ricochet, obstacle_type, seeded, o_switch, use_checkpoint, decay_factor, pheromone_receptor, collision_threshold, time_constant, time_bonus_multiplier, teleport, num_sensors, fitness_criterion, food_calibration, endless, SPARSE_REWARD, stagnation, NUM_RUNS
    stagnation = args.stagnation
    extra_sparse = str2bool(args.extra_sparse)
    SPARSE_REWARD = str2bool(args.sparse_reward)
    num_sensors = args.num_sensors
    endless = str2bool(args.endless)
    fitness_criterion = args.fitness_criterion
    food_calibration = str2bool(args.food_calibration)
    pheromone_receptor = str2bool(args.pheromone_receptor)
    teleport = str2bool(args.teleport)
    # Set global variables based on parsed arguments
    decay_factor = args.decay_factor
    use_checkpoint = args.use_checkpoint
    seeded = str2bool(args.seeded)
    obstacle_type = args.obstacle_type
    obstacles = str2bool(args.obstacles)
    particles = args.particles
    generations = args.generations
    movement_type = args.movement_type
    network_type = args.network
    o_switch = str2bool(args.orientation_switching)
    sub = args.sub
    test_run = str2bool(args.test)
    ricochet = str2bool(args.ricochet)
    best_file = args.best
    collision_threshold = args.collision_threshold
    time_constant = args.time_constant
    time_bonus_multiplier = args.time_bonus_multiplier

    manual_testing(config)