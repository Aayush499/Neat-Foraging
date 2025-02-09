import pickle
import neat
import graphviz
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def load_genome(GENOME_FILE):
    with open(GENOME_FILE, "rb") as f:
        genome = pickle.load(f)
    return genome

def genome_to_xml(genome, genome_id, config):
    print("Input: ", config.genome_config.input_keys)
    root = ET.Element('chromosome', id = str(genome_id))

    #Add input nodes manually
    #In the tool-use case, there are 9 inputs
    #The default config.genome_config.input_keys represent the input nodes as negative integers
    #Change them into positive int
    for input_id in config.genome_config.input_keys:
        neuron_elem = ET.SubElement(root, 'neuron', id = str(-input_id), type = 'in', activation = 'linear')

    for node_id, node in genome.nodes.items():
        #Manually change output node numbering
        #Since the output nodes are 0 and 1, they interfere with the output node ids
        nodeType = node_type(node_id, config)
        if nodeType == "out":
            neuron_elem = ET.SubElement(root, 'neuron', id = "output" + str(node_id), type = nodeType, activation = node_activation(node))
        else:
            neuron_elem = ET.SubElement(root, 'neuron', id = str(node_id), type = nodeType, activation = node_activation(node))

    for conn_id, conn in genome.connections.items():
        src = conn.key[0]
        dest = conn.key[1]
        if src == 0:
            src = "output0"
        elif src == 1:
            src = "output1"
        elif src < 0:
            src *= -1
        src = str(src)

        if dest == 0:
            dest = "output0"
        elif dest == 1:
            dest = "output1"
        elif dest < 0:
            dest *= -1
        dest = str(dest)

        id_src, id_dest = conn_id
        if id_src == 0:
            id_src = "output0"
        elif id_src == 1:
            id_src = "output1"
        elif id_src < 0:
            id_src *= -1

        if id_dest == 0:
            id_dest = "output0"
        elif id_dest == 1:
            id_dest = "output1"
        elif id_dest < 0:
            id_dest *= -1
        
        conn_id = "(" + str(id_src) + ", " + str(id_dest) + ")"

        connection_elem = ET.SubElement(root, 'connection', 
                                        id = conn_id,
                                        **{'src-id': src, 'dest-id': dest},
                                        weight = str(conn.weight))
    
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent = "")

    pretty_xml_without_declaration = "\n".join(pretty_xml.split("\n")[1:])

    return pretty_xml_without_declaration


def node_type(node_id, config):
    if node_id in config.genome_config.input_keys:
        return "in"
    elif node_id in config.genome_config.output_keys:
        return "out"
    else:
        return "hidden"

def node_activation(node):
    return node.activation

if __name__ == "__main__":
    GENOME_FILE = "db/genomes/29062.pkl"
    genome = load_genome(GENOME_FILE)

    config_file = 'config-tooluse.txt'
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    genome_id = genome.key
    xml_output = genome_to_xml(genome, genome_id, config)

    with open(f'chromosome{genome_id}.xml', 'w') as f:
        f.write(xml_output)
