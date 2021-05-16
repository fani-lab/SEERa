import networkx as nx
import os
import glob
import matplotlib.pyplot as plt



def EmbeddedGraphShow(G,day):
    G = G.subgraph(list(G.nodes)[:200])
    nx.draw(G)#, with_labels=True)
    plt.interactive(False)
    # plt.show(block=True)
    plt.savefig('Embedding'+str(day+1)+'.jpg')
    plt.close()

def GraphShow(G,day):
    G = G.subgraph(list(G.nodes)[:200])
    nx.draw(G)#, with_labels=True)
    plt.interactive(False)
    # plt.show(block=True)
    plt.savefig('Graph'+str(day)+'.jpg')
    plt.close()

run_list = glob.glob('run_outputs/2021*')
print(run_list[-1])
os.chdir(run_list[-1]+'/graphs')

graphs_path = glob.glob('*.net')
graphs = []
en = 1
for gp in graphs_path:
    g = nx.read_gpickle(gp)
    print('Graph:', gp, 'has', len(g.nodes), 'nodes.')
    GraphShow(g, en)
    en += 1
