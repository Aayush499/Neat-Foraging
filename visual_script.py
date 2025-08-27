import neat
import pickle
import os

# If you have the 'visualize.py' script from NEAT-Python examples, import it.
# If not, download it and place it in your project directory.
import visualize

# Load config and genome.
config_file = 'config-replication-unstable'
genome_file = 'best.pickle'

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_file
)

with open(genome_file, "rb") as f:
    genome = pickle.load(f)

# node_names is optional: map node numbers to labels, e.g. {-1: "in1", 0: "out"}
node_names = None  # You can define as dict if you want

# Draw the network.
visualize.draw_net(config, genome, view=True, node_names=node_names, prune_unused=True)
