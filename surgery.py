import neat
import pickle
#load  best.pickl
# import neat
import pickle
import networkx as nx

import argparse
import subprocess

from Forage import Game
import pygame
import neat
import os
import time
import pickle
import matplotlib.pyplot as plt
from Forage.food import Food
import numpy as np
import glob
import configparser
import csv
import tempfile
import math

#make current working directory as the directory to look for the genome file

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def arg_parser():
    parser = argparse.ArgumentParser(description="Surgery on NEAT genomes.")
    parser.add_argument("--genome_file", type=str, help="Path to the genome pickle file.")
    parser.add_argument("--output_file", type=str, default="modified_genome.pkl",
                        help="Path to save the modified genome pickle file.")
    parser.add_argument("--source_node", type=int, help="Node ID to remove connections from.")
    parser.add_argument("--target_node", type=int, help="Node ID to remove connections to.")
    return parser

#for now, just print the genome_file path


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    genome_file = args.genome_file
    output_file = args.output_file

    genome_file = 'best_genome_gen_323.pkl'
    print(f"Loading genome from: {genome_file}")
    
    # Load the genome
    with open(genome_file, "rb") as f:
        genome = pickle.load(f)
    
    # Perform surgery: Example connection between source_node and target_node
    #disable connectgion between source_node and target_node
    source_node = args.source_node
    target_node = args.target_node
    for conn_key in list(genome.connections.keys()):
        conn = genome.connections[conn_key]
        if conn_key[0] == source_node and conn_key[1] == target_node:
            print(f"Disabling connection from {source_node} to {target_node}")
            conn.enabled = False

    # Save the modified genome
    with open(output_file, "wb") as f:
        pickle.dump(genome, f)

    print(f"Modified genome saved to: {output_file}")