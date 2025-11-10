import neat
import pickle
import os
import configparser
import tempfile
# If you have the 'visualize.py' script from NEAT-Python examples, import it.
# If not, download it and place it in your project directory.
import visualize
import math
import networkx as nx


pheromone_receptor = False
network_type = 'recursive'  # Default value


# Load config and genome.
config_path = 'configs/config-default'
genome_file = 'best_networks/aayush_baby_3.pickle'

cfg = configparser.ConfigParser()
cfg.read(config_path)


#get num_sensors value from the genome_file, the number ater 'NS'
# filename = os.path.basename(genome_file)
# ns_index = filename.find('-NS')
# num_sensors_str = ''
# for char in filename[ns_index + 3:]:
#     if char.isdigit():
#         num_sensors_str += char
#     else:
#         break
# num_sensors = int(num_sensors_str) if num_sensors_str else 0

#get num_sensors from the input_number in the config file
num_sensors = int(cfg['DefaultGenome']['num_inputs'])

#get pheromone receptor value from the genome_file, the value after 'P'
# p_index = filename.find('-P')
# pheromone_receptor_str = ''
# for char in filename[p_index + 2:]:
#     if char in ['T', 'F']:
#         pheromone_receptor_str += char
#         break
#     else:
#         break




input_size = num_sensors * (1 + int(pheromone_receptor))
hidden_size = math.ceil(input_size *3/4)

#get the network type from the genome_file, the value after -N
network_type = 'recursive'  # Default value
# n_index = filename.find('-N')
# if n_index != -1:
#     network_type = filename[n_index + 2:].split('-')[0]
cfg['DefaultGenome']['num_inputs'] = str(input_size)
cfg['DefaultGenome']['num_hidden'] = str(hidden_size)
#outputs
cfg['DefaultGenome']['num_outputs'] = '3'
if network_type == 'ff':
    cfg['DefaultGenome']['feed_forward'] = 'True'
else:
    cfg['DefaultGenome']['feed_forward'] = 'False'

cfg['NEAT']['fitness_criterion'] = 'mean'

config = None
# 3. Write to temporary file and use with NEAT
with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
    cfg.write(tmpfile)
    tmpfile.flush()  # ensure data is written

    # 4. Create NEAT config with temp file
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        tmpfile.name
    )




with open(genome_file, "rb") as f:
    genome = pickle.load(f)

# node_names is optional: map node numbers to labels, e.g. {-1: "in1", 0: "out"}
node_names = None  # You can define as dict if you want



# Extract enabled connections
connections = [(conn.key[0], conn.key[1]) for conn in genome.connections.values() if conn.enabled]

# Build directed graph
G = nx.DiGraph()
G.add_edges_from(connections)

# Find all cycles (recursive connections)
cycles = list(nx.simple_cycles(G))
nodes_in_cycles = set()
for cycle in cycles:
    nodes_in_cycles.update(cycle)

special_node_colors = {}
for n in nodes_in_cycles:
    special_node_colors[n] = 'red'  # Color for nodes involved in cycles
# Draw the network.
dot = visualize.draw_net(config, genome, view=True, node_names=node_names, prune_unused=False, node_colors=special_node_colors)

print("hello")
