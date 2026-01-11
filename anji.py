#!/usr/bin/python
"""
anji2dot.py : convert NEAT chromosome XML file to Graphviz DOT format,
              with cycle detection and highlighting.

Usage:
    anji2dot.py <chromosome.xml> [--out test.dot] [--show-weights] [--max-cycles 10]
"""

import sys
import math
import xml.etree.ElementTree as ET
import graphviz
import argparse

try:
    import networkx as nx
except ImportError:
    nx = None
    print("[warning] networkx not found, cycle detection disabled.")

#----------------------------------------
def parse_xml(xmlfile):
#----------------------------------------
    tree = ET.parse(xmlfile)
    root = tree.getroot()

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

#----------------------------------------
def detect_cycles(nodes, edges, max_cycles=None):
#----------------------------------------
    if nx is None:
        return []

    G = nx.DiGraph()
    for nid in nodes:
        G.add_node(nid)
    for e in edges:
        G.add_edge(e['src'], e['dst'], weight=e['weight'])

    cycles = list(nx.simple_cycles(G))
    if max_cycles:
        cycles = cycles[:max_cycles]
    return cycles

#----------------------------------------
def build_dot(nodes, edges, cycles, show_weights=False):
#----------------------------------------
    ret = "digraph G {\n"
    ret += "rankdir=BT;\n"

    # --- Node rendering ---
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

    # --- Edge rendering ---
    cyc_edges = set()
    for cyc in cycles:
        cyc_edges |= {(cyc[i], cyc[(i+1)%len(cyc)]) for i in range(len(cyc))}

    for e in edges:
        style = "bold" if e['weight'] > 0 else "dashed"
        color = "blue" if e['weight'] > 0 else "red"
        penwidth = 1.0 + 1.5 * min(abs(e['weight']) / 5.0, 1.0)

        if (e['src'], e['dst']) in cyc_edges:
            style = "bold"
            color = "purple"

        label = f'label="{e["weight"]:.2f}"' if show_weights else ""
        ret += f'n{e["src"]} -> n{e["dst"]} [color="{color}", style="{style}", penwidth="{penwidth:.2f}", {label}];\n'

    # --- Rank grouping ---
    ins = " ".join(f"n{n}" for n,t in nodes.items() if t == "in")
    outs = " ".join(f"n{n}" for n,t in nodes.items() if t == "out")
    ret += f"{{rank=min; {ins}}}\n"
    ret += f"{{rank=max; {outs}}}\n"
    ret += "}\n"
    return ret

#----------------------------------------
def main():
#----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("xmlfile", help="NEAT chromosome XML file")
    parser.add_argument("--out", default="test.dot", help="Output DOT filename")
    parser.add_argument("--show-weights", action="store_true", help="Show weights on edges")
    parser.add_argument("--max-cycles", type=int, default=None, help="Limit number of cycles visualized")
    args = parser.parse_args()

    nodes, edges = parse_xml(args.xmlfile)
    cycles = detect_cycles(nodes, edges, args.max_cycles)
    if cycles:
        print(f"Detected {len(cycles)} cycles:")
        for i, c in enumerate(cycles):
            print(f"  cycle {i+1}: {' → '.join(c)}")
    else:
        print("No cycles detected.")

    dot_str = build_dot(nodes, edges, cycles, show_weights=args.show_weights)
    with open(args.out, "w") as f:
        f.write(dot_str)

    s = graphviz.Source(dot_str)
    s.render(args.out, format="pdf", cleanup=True)
    print(f"Saved to {args.out} and {args.out}.pdf")

if __name__ == "__main__":
    main()
