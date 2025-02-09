#!/usr/bin/python

'''

  anji2dot.py : convert NEAT chromosome XML file to graphviz DOT format.

  usage: 

     anji2dot.py <chromosome.xml> 

  results:

     test.dot : DOT file
     test.dot.pdf : PDF file

'''

import sys
import graphviz
import xml.etree.ElementTree as ET

#----------------------------------------
def parse_XML(xmlfile):
#----------------------------------------
  '''
  parseXML : load xml file and parse it
  
  arguments: 
    xmlfile  : XML filename

  returns:
    DOT format string for nodes and edges 
  '''

  tree = ET.parse(xmlfile)
  root = tree.getroot()
  ret  = ""

  in_count = 1
  out_count = 1

  # 1. process neurons
  for item in root.findall('neuron'):
    
      n_id   = item.get('id')
      n_type = item.get('type')

      # if input node
      if (n_type == "in"):
        ret = ret+"n{} [shape=\"box\" label=\"I_{}\", fillcolor=\"red\" style=\"filled\", pos=\"{},6.5!\"];\n".\
	              format(str(n_id),str(n_id),str(in_count*1.5))
        in_count += 1
      elif (n_type == "out"):
        ret = ret+"n{} [shape=\"circle\" label=\"O_{}\", fillcolor=\"green\" style=\"filled\", pos=\"{},0.5!\"];\n".\
	              format(str(n_id),str(n_id),str(out_count*1.5))
        out_count += 1
      else:
        ret = ret+"n{} [shape=\"hexagon\", label=\"h_{}\", fillcolor=\"orange\" style=\"filled\"];\n".\
	              format(str(n_id),str(n_id))

  # 2. process connections
  for item in root.findall('connection'):

      n_id    = item.get('id')
      src_id  = item.get('src-id')
      dest_id = item.get('dest-id')
      weight  = item.get('weight')

      if (float(weight)>0):
        ret = ret+"n{} -> n{} [style={}, arrowsize={}, arrowhead={}];\n".format(src_id,dest_id,"bold","2.0","normal")
      else:
        ret = ret+"n{} -> n{} [style={}, arrowsize={}, arrowhead={}];\n".format(src_id,dest_id,"dashed","1.0","dot")

  return ret

#----------------------------------------
def get_neuron_by_type(xmlfile,n_type):
#----------------------------------------

  tree = ET.parse(xmlfile)
  root = tree.getroot()

  ret  = ""

  # 1. process neurons
  for item in root.findall('neuron'):
    
    n_id = item.get('id')
    n_ty  = item.get('type')
 
    if (n_ty == n_type):
      ret = ret + "\"n" + n_id + "\";"
    
  return ret

#----------------------------------------
def print_anji(xmlfile):
#----------------------------------------
  '''
  printAnji : print out DOT file
  
  arguments:
    xmlfile : XML filename
  '''

  ret = "digraph G {\n";
  ret = ret + "rankdir=BT;\n"
  ret = ret + parse_XML(xmlfile)
  ret = ret + "{rank = min;" + get_neuron_by_type(xmlfile,"in") + "}\n"
  ret = ret + "{rank = max;" + get_neuron_by_type(xmlfile,"out") + "}\n"
  ret = ret + "}"
  return ret

#----------------------------------------
# Test 
#----------------------------------------

if len(sys.argv)==1:
  print("usage: anji2dot.py <chromosome.xml>");
  exit()

orig_stdout = sys.stdout 
 
print("Saving to test.dot")
with open('test.dot','w') as f:
  sys.stdout = f
  print(print_anji(sys.argv[1]))
  sys.stdout = orig_stdout

s=graphviz.Source.from_file('test.dot')
s.view()
