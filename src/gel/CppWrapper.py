import os
from gel import GraphToText as GTT

def main(graphPath, newGraphPath, dim):
    GTT.G2T(graphPath, newGraphPath)
    # os.chdir(graphPath)#'source_code/main')
    os.chdir('gel/source_code/main')
    stream = os.popen(f'.\BCGDEmbed ../../../{newGraphPath} -c {dim}')
    output = stream.read()
    print(output)
    GTT.T2A('../main', '../../../'+newGraphPath)