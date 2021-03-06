import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools


def generate_graph(nodes, plot=False):
    """
    Function to generate the graph from the nodes (instances)

    Args:
        nodes (int): number of nodes (instances) at each frame
        plot (bool): choose if plot or not the graph once created
    Returns:
        G (nx Graph): graph with the nodes corresponding to the instances in each frame
        pos (dict): a dictionary of positions keyed by node.
    """
    result = itertools.accumulate(nodes)
    prev_value = 0
    layers = []
    for value in result:
        layers.append(range(prev_value, value))
        prev_value = value
    G = nx.DiGraph()
    for (i, layer) in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    pos = nx.multipartite_layout(G, subset_key="layer")

    if plot:
        plt.axis("auto")
        plt.show()

    return G, pos, layers


def filter_graph(G):
    """
    Function to filter edges and nodes in the graph, to avoid the same node to split
    in two different nodes and to avoid multiple edges to converge on a single node. The
    strategy of keeping the highest weight is teken into account

    Args:
        G (nx Graph): graph to be filtered
    Returns:
        G (nx Graph): cleaned graph
    """
    nodes = G.nodes()
    for node in nodes:
        in_edges = G.in_edges(node, data=True)
        weight = 0
        u_max = None
        v_max = None
        for edge in in_edges:
            u = edge[0]
            v = edge[1]
            # The current weight is bigger than provious one
            if G[u][v]['weight'] >= weight:
                weight = G[u][v]['weight']
                if u_max and v_max:
                    G.remove_edge(u_max, v_max)
                u_max = u
                v_max = v
            # The current weight is bigger than provious one
            elif G[u][v]['weight'] <= weight:
                G.remove_edge(u, v)
        out_edges = G.out_edges(node, data=True)
        u_max = None
        v_max = None
        weight = 0
        for edge in list(out_edges):
            u = edge[0]
            v = edge[1]
            # The current weight is bigger than provious one
            if G[u][v]['weight'] > weight:
                weight = G[u][v]['weight']
                if u_max and v_max:
                    G.remove_edge(u_max, v_max)
                u_max = u
                v_max = v
            # The current weight is bigger than provious one
            elif G[u][v]['weight'] <= weight:
                G.remove_edge(u, v)
    return G

def explore_edge(G, node, path, banned):
    """
    Function to extract the paths of all nodes in the graph

    Args:
        G (nx Graph): Graph to extract paths
        path (list): list with nodes of the current path to explore
        node (nx Node): choose if plot or not the graph once created
        banned (list): list of the nodes visited
    Returns:
        path (list): list containing the nodes visited in this path
        banned (list): list containind the visited nodes
    """
    out_edge = G.out_edges(node, data=True)
    edge = list(out_edge)
    if edge:
        for value in edge:
            u = value[0]
            v = value[1]
            path.append(u)
            banned.append(u)
            explore_edge(G, v, path, banned)
    else:
        path.append(node)
        banned.append(node)
    return path, banned