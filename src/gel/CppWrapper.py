import glob
import os
from gel import GraphToText as GTT

def main(graphPath, newGraphPath, dim):
    GTT.G2T(graphPath, newGraphPath)
    os.chdir('gel/Temporal-Network-Embedding/source_code/main')
    script = f'.\BCGDEmbed ../../../../{newGraphPath} -c {dim}'
    stream = os.popen(script)
    output = stream.read()
    print(output)
    GTT.T2A('../main', '../../../../'+newGraphPath)
    os.chdir('../../../../')