import os
import networkx as nx
import glob
import numpy as np
import params

def G2T(graph_path, path2_save_gel=params.uml['path2saveGEL']):
    graphs_path = glob.glob(f'{graph_path}/*.net')
    for gp in graphs_path:
        pathtemp = path2_save_gel+'/' + str(gp.split("\\")[-1].split(".")[0])
        if os.path.exists(pathtemp + '.txt'):
            os.remove(pathtemp + '.txt')
        File_object = open(pathtemp + '.txt', 'a+')
        g = nx.read_gpickle(gp)
        File_object.write(str(len(g.nodes())))
        File_object.write('\n')
        for i in range(g.order()):
            File_object.write(str(i))
            File_object.write(',')
            File_object.write(str(len(g.edges([i]))-1))
            edges = g.edges([i])
            for edge in edges:
                if edge[0] != edge[1]:
                    File_object.write(':')
                    File_object.write(str(edge[1]))
                    File_object.write(',')
                    File_object.write(str(g.get_edge_data(i, edge[1])['weight'].__round__()))
            File_object.write('\n')
        File_object.close()


def T2A(path2read, path2save):
    embeddedspath = glob.glob(f'{path2read}/Zmatrix*')
    for embeddedpath in embeddedspath:
        file1 = open(embeddedpath, 'r')
        lines = file1.readlines()
        array = np.zeros((int(lines[0].split('\n')[0]), params.uml['EmbeddingDim']))
        for i in range(1,len(lines)):
            line = lines[i]
            parts = line.split(':')
            for j in range(1, len(parts)):
                array[i-1, int(parts[j].split(',')[0])] = float(parts[j].split(',')[1])
        pathtemp = embeddedpath.split('\\')[-1]
        np.save(f'{path2save}/{pathtemp}.npy', array)
    npys = glob.glob(f'{path2save}/*.npy')
    t = []
    for tt in npys:
        t.append(np.load(tt))
    np.save(f'{path2save}/embeddeds.npy', array)
