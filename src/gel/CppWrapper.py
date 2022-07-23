import glob
import os
from gel import GraphToText as GTT

def main(graphPath, new_graph_path, dim):
    GTT.g2t(graphPath, new_graph_path)
    os.chdir('gel/Temporal-Network-Embedding/source_code/main')
    script = f'.\BCGDEmbed ../../../../{new_graph_path} -c {dim}'
    stream = os.popen(script)
    output = stream.read()
    GTT.t2a('../../../../'+new_graph_path, '../../../../'+new_graph_path)
    os.chdir('../../../../')