import os
import glob
import pickle
import re
import math
import tempfile
import configparser
import neat
import xml.etree.ElementTree as ET
import graphviz
import networkx as nx
import argparse
# =================
# ---- USER SETTINGS ----
genome_dir = 'checkpoints/checkpoint-OFalse-F3-holonomic-G700-Nrecursive-Srecursivve_test__sparse_2-RFalse-OTline-SEFalse-OSTrue-D0.97-PFalse-CT10-TC450.0-TBonus2.0-TFalse-NS8-FCTrue-Fmax-St30/best_genomes_by_generation'
out_img_format = "svg"  # 'svg', 'png', or 'pdf'
default_param = True
num_sensors = 8
pheromone_receptor = 0  # set 0 or 1
movement_type = 'holonomic'  # or 'holonomic'
network_type = 'recurrent'  # or 'recurrent'
SPARSE_REWARD = True
stagnation = 50
# =================


# ---- Dynamic Config Synthesis ----
local_dir = os.path.dirname(__file__)
config_filename = 'config-default' if default_param else 'config-simple'
config_path = os.path.join(local_dir, 'configs', config_filename)
print("Using config file:", config_path)
cfg = configparser.ConfigParser()
cfg.read(config_path)

input_size = num_sensors * (1 + int(pheromone_receptor))
hidden_size = math.ceil(input_size * 3 / 4)
output_size = 1
if movement_type == "holonomic":
    output_size += 1
if pheromone_receptor:
    output_size += 1

cfg['DefaultGenome']['num_inputs'] = str(input_size)
cfg['DefaultGenome']['num_hidden'] = str(hidden_size)
cfg['DefaultGenome']['num_outputs'] = str(output_size)
cfg['DefaultGenome']['feed_forward'] = 'True' if network_type == 'ff' else 'False'
cfg['DefaultStagnation']['max_stagnation'] = str(stagnation)
cfg['NEAT']['fitness_criterion'] = 'max'

if SPARSE_REWARD:
    cfg['NEAT']['fitness_threshold'] = '20'
else:
    cfg['NEAT']['fitness_threshold'] = '700'

# Write config to temp file for NEAT
with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
    cfg.write(tmpfile)
    tmpfile.flush()
    neat_config_path = tmpfile.name

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    neat_config_path
)

# ---- Visualization Code ----
def esc_str(s):
    s = "n" + str(s)
    return re.sub("-", "_", s)

def get_xml(genome, config):
    nn = neat.nn.RecurrentNetwork.create(genome, config)
    nodes = genome.nodes
    connections = genome.connections
    input_nodes = nn.input_nodes
    output_nodes = nn.output_nodes
    xml = '<chromosome id="0" primary-parent-id="0">\n'
    for input_node in input_nodes:
        input_node = esc_str(input_node)
        xml += f'<neuron id="{input_node}" type="in" activation="clamped"/>\n'
    for key in nodes:
        node = nodes[key]
        t = "out" if key in output_nodes else "hid"
        key = esc_str(key)
        xml += f'<neuron id="{key}" type="{t}" activation="{node.activation}"/>\n'
    for (src, dest) in connections:
        connection = connections[(src, dest)]
        if connection.enabled:
            src = esc_str(src)
            dest = esc_str(dest)
            xml += f'<connection id="0" src-id="{src}" dest-id="{dest}" weight="{connection.weight}"/>\n'
    xml += '</chromosome>'
    return xml

def parse_xml_str(xmlstring):
    root = ET.fromstring(xmlstring)
    nodes, edges = {}, []
    for n in root.findall('neuron'):
        nid = n.get('id')
        ntype = n.get('type')
        nodes[nid] = ntype
    for c in root.findall('connection'):
        edges.append({
            "src": c.get('src-id'),
            "dst": c.get('dest-id'),
            "weight": float(c.get('weight', '0')),
            "id": c.get('id')
        })
    return nodes, edges

def detect_cycles_nodes_edges(nodes, edges, max_cycles=None):
    G = nx.DiGraph()
    for nid in nodes:
        G.add_node(nid)
    for e in edges:
        G.add_edge(e['src'], e['dst'], weight=e['weight'])
    cycles = list(nx.simple_cycles(G))
    if max_cycles:
        cycles = cycles[:max_cycles]
    return cycles

def build_dot(nodes, edges, cycles, show_weights=False):
    ret = "digraph G {\nrankdir=BT;\n"
    in_count = out_count = 1
    cyc_nodes = set(n for cyc in cycles for n in cyc)
    for nid, ntype in nodes.items():
        if ntype == "in":
            ret += f'n{nid} [shape="box" label="I_{nid}", fillcolor="red" style="filled", pos="{in_count*1.5},6.5!"];\n'
            in_count += 1
        elif ntype == "out":
            ret += f'n{nid} [shape="circle" label="O_{nid}", fillcolor="green" style="filled", pos="{out_count*1.5},0.5!"];\n'
            out_count += 1
        else:
            color = "orange" if nid not in cyc_nodes else "yellow"
            ret += f'n{nid} [shape="hexagon", label="h_{nid}", fillcolor="{color}" style="filled"];\n'
    cyc_edges = set((cyc[i], cyc[(i+1)%len(cyc)]) for cyc in cycles for i in range(len(cyc)))
    for e in edges:
        style = "bold" if e['weight'] > 0 else "dashed"
        color = "blue" if e['weight'] > 0 else "red"
        penwidth = 1.0 + 1.5 * min(abs(e['weight']) / 5.0, 1.0)
        if (e['src'], e['dst']) in cyc_edges:
            style = "bold"
            color = "purple"
        label = f'label="{e["weight"]:.2f}"' if show_weights else ""
        ret += f'n{e["src"]} -> n{e["dst"]} [color="{color}", style="{style}", penwidth="{penwidth:.2f}", {label}];\n'
    ins = " ".join(f"n{n}" for n, t in nodes.items() if t == "in")
    outs = " ".join(f"n{n}" for n, t in nodes.items() if t == "out")
    ret += f"{{rank=min; {ins}}}\n"
    ret += f"{{rank=max; {outs}}}\n"
    ret += "}\n"
    return ret

def visualize_genomes_in_dir(genome_dir, config, out_img_format="svg"):
    # Make output directory
    out_img_dir = os.path.join(genome_dir, "visualizations")
    os.makedirs(out_img_dir, exist_ok=True)
    genome_files = sorted(glob.glob(os.path.join(genome_dir, "best_genome_gen_*.pickle")))
    for genome_file in genome_files:
        with open(genome_file, "rb") as f:
            genome = pickle.load(f)
        xml_str = get_xml(genome, config)
        nodes, edges = parse_xml_str(xml_str)
        cycles = detect_cycles_nodes_edges(nodes, edges)
        dot_str = build_dot(nodes, edges, cycles, show_weights=True)
        dot_name = os.path.splitext(os.path.basename(genome_file))[0]
        out_path = os.path.join(out_img_dir, f"{dot_name}.{out_img_format}")
        # Render directly, don't keep DOT or XML files, just image.
        graphviz.Source(dot_str).render(out_path, format=out_img_format, cleanup=True)
        print(f"Visualized {genome_file} as {out_path}")


def args_parser():
    
    parser = argparse.ArgumentParser(description="Visualize NEAT genomes in a directory.")
    parser.add_argument("--genome_dir", type=str, help="Directory containing genome pickle files.")
    parser.add_argument("--out_img_format", type=str, default="svg", choices=["svg", "png", "pdf"],
                        help="Output image format for visualizations.")
    parser.add_argument("--whole", help="Visualize whole directory or specific file.", type=bool, default=False)
    parser.add_argument("--genome_file", type=str, help="Path to a specific genome pickle file to visualize.", default=None)

    return parser

if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()
    genome_dir = args.genome_dir
    out_img_format = args.out_img_format
    whole = args.whole
    genome_file = args.genome_file
    concatenate_path = os.path.join(os.getcwd(), genome_dir)
    genome_dir = concatenate_path
    concatenate_path = os.path.join(genome_dir, genome_file)
    if whole:
        visualize_genomes_in_dir(genome_dir, config, out_img_format=out_img_format)
    else:
        #visualize a specific genome file in the current directory (don't save image)
        with open(concatenate_path, "rb") as f:
            genome = pickle.load(f)
        xml_str = get_xml(genome, config)
        nodes, edges = parse_xml_str(xml_str)
        cycles = detect_cycles_nodes_edges(nodes, edges)
        dot_str = build_dot(nodes, edges, cycles, show_weights=True)
        graph = graphviz.Source(dot_str)
        graph.render(view=True, format=out_img_format, cleanup=True)

        
     
       