''' 
Author: Minyoung Kim ( Jul. 1, 2022)
2022 Hackaton 
Team: ThankQ
description: 
    generate nx.graph from data and distance functions 
''' 
import networkx as nx
import numpy as np 


def get_example_graph():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    nx.draw(graph, with_labels=True, alpha=0.8, node_size=500)
    return graph


def generate_graph(data, distftn, draw=True):
    node_num, feature_num = data.shape 
    #Generate adjacency matrix using distance function. (i,j) component indicates distance between two nodes. 
    adjacency_mat = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            if i==j: continue
            else: adjacency_mat[i,j]+=distftn(data[i], data[j])
    graph = generate_graph_from_numpy(adjacency_mat, draw = draw)
    return graph 
    
def generate_graph_from_numpy(adjacency_mat, draw = True):
    graph = nx.from_numpy_matrix(adjacency_mat)
    if draw: 
        layout = nx.random_layout(graph, seed = 10 )
        labels = dict([((u,v,), f"{d['weight']:.3f}") for u,v,d in graph.edges(data=True)])
        #labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw(graph, layout, with_labels=True)
        nx.draw_networkx_edge_labels(graph,pos=layout, edge_labels=labels)
    return graph

# sample_data = np.ones((2, 3)) #data # = 2, 3 features 
# cosftn = lambda x, y : np.dot(x, y)

# generate_graph(sample_data, cosftn)