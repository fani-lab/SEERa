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

def GraphShow(G,day,RunId):
    G = G.subgraph(list(G.nodes)[:200])
    nx.draw(G)#, with_labels=True)
    plt.interactive(False)
    # plt.show(block=True)
    plt.savefig(f'../output/{RunId}/uml/graphs/Graph{str(day)}.jpg')
    plt.close()

def main(RunId):
    print(os.getcwd())
    graphs_path = glob.glob(f'../output/{RunId}/uml/graphs/*.net')
    print(graphs_path)
    en = 1
    for gp in graphs_path:
        g = nx.read_gpickle(gp)
        print('Graph:', gp, 'has', len(g.nodes), 'nodes.')
        GraphShow(g, en, RunId)
        en += 1

    # graphs_path = glob.glob(f'../output/{RunId}/uml/graphs/pajek/*.net')
    # print(graphs_path)
    # en = 1
    # for gp in graphs_path:
    #     g = nx.read_pajek(gp)
    #     print('Graph:', gp, 'has', len(g.nodes), 'nodes.')
    #     GraphShow(g, en, RunId)
    #     en += 1
