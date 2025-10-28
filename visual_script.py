import neat
import pickle
import os
import configparser
import tempfile
# If you have the 'visualize.py' script from NEAT-Python examples, import it.
# If not, download it and place it in your project directory.
import visualize
import math

# Load config and genome.
config_path = 'configs/config-simple'
genome_file = 'best_networks/best-OFalse-F3-holonomic-G700-Nrecursive-Sdefault_params_test1-RFalse-OTline-SEFalse-OSFalse-D0.97-PFalse-CT7-TC600.0-TBonus4.0-TFalse-NS8-FCTrue-Fmax.pickle'

cfg = configparser.ConfigParser()
cfg.read(config_path)


#get num_sensors value from the genome_file, the number ater 'NS'
filename = os.path.basename(genome_file)
ns_index = filename.find('-NS')
num_sensors_str = ''
for char in filename[ns_index + 3:]:
    if char.isdigit():
        num_sensors_str += char
    else:
        break
num_sensors = int(num_sensors_str) if num_sensors_str else 0
#get pheromone receptor value from the genome_file, the value after 'P'
p_index = filename.find('-P')
pheromone_receptor_str = ''
for char in filename[p_index + 2:]:
    if char in ['T', 'F']:
        pheromone_receptor_str += char
        break
    else:
        break
pheromone_receptor = pheromone_receptor_str == 'T'



input_size = num_sensors * (1 + int(pheromone_receptor))
hidden_size = math.ceil(input_size *3/4)

#get the network type from the genome_file, the value after -N
network_type = 'recursive'  # Default value
n_index = filename.find('-N')
if n_index != -1:
    network_type = filename[n_index + 2:].split('-')[0]
cfg['DefaultGenome']['num_inputs'] = str(input_size)
cfg['DefaultGenome']['num_hidden'] = str(hidden_size)
if network_type == 'ff':
    cfg['DefaultGenome']['feed_forward'] = 'True'
else:
    cfg['DefaultGenome']['feed_forward'] = 'False'

# cfg['NEAT']['fitness_criterion'] = 'mean'

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

# Draw the network.
dot = visualize.draw_net(config, genome, view=True, node_names=node_names, prune_unused=False)

print("hello")
