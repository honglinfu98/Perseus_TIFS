"""
This algorithm is used to create netwrok of the channeles and based on the paper below:
DANI: A Fast Diffusion Aware Network Inference Algorithm
"""

from collections import defaultdict
import numpy as np
import networkx as nx


def sort_cascade(cascade: list):
    """
    This function is used to sort the cascade based on the infected time
    Return: sorted cascade
    """
    node_id = np.array(cascade[0])
    infected_time = np.array(cascade[1])
    index = np.argsort(infected_time)
    sorted_node_id = node_id[index].tolist()
    sorted_infected_time = infected_time[index].tolist()
    return [sorted_node_id, sorted_infected_time]


def trans_list(cascade_dict: dict):
    """
    This function is used to transform the cascade from dictionary to list
    Return: cascade in list format
    """
    cascade = []
    nodes = []
    times = []

    for node in cascade_dict.keys():
        if node != "T":
            nodes.append(int(node))
            times.append(cascade_dict[node])

    cascade.append(nodes)
    cascade.append(times)

    return cascade


def get_index(cascade: list):
    """
    This function is used to get the index of the nodes in the cascade
    Return: index of the nodes in the cascade
    """
    S = sort_cascade(cascade)
    CV = {}
    nodes = S[0]
    for i, node in enumerate(nodes):
        CV[node] = i + 1

    return CV


def DANI(N: int, cascades: list):
    """
    This script is used to process the cascades and return the influence graph
    Return: Influence graph, edge list sorted, , P_dict
    """
    P = np.zeros([N, N])

    for cascade_dict in cascades:
        D = np.zeros([N, N])
        cascade = trans_list(cascade_dict)
        S = sort_cascade(cascade)
        CV = get_index(cascade)
        nodes = S[0]

        for i in range(len(nodes)):
            u = nodes[i]
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                D[u][v] = 1 / (CV[v] * (CV[v] - CV[u]))

        for i in range(N):
            D[i] = D[i] / np.sum(D[i])
            for j in range(N):
                if D[i][j] > 0:
                    P[(i, j)] += D[i][j]

    P_dict = defaultdict(float)
    for i in range(N):
        P[i] = P[i] / np.sum(P[i])
        for j in range(N):
            if i == j:
                continue
            else:
                if P[i][j] > 0:
                    P_dict[(i, j)] = P[i][j]

    node_status_set = {}
    for i in range(N):
        node_status_set[i] = set()
        for c_i, cascade_dict in enumerate(cascades):
            cascade = trans_list(cascade_dict)
            if i in cascade[0]:
                node_status_set[i].add(c_i)

    A = {}
    for u in range(N):
        for v in range(u + 1, N):
            A[(u, v)] = (
                len(node_status_set[u] & node_status_set[v])
                / len(node_status_set[u] | node_status_set[v])
                * (P_dict[(u, v)] + P_dict[(v, u)])
            )

    result = []
    for key in A.keys():
        result.append((key, A[key]))

    def takeSecond(elem):
        return elem[1]

    result.sort(key=takeSecond, reverse=True)

    IG = nx.DiGraph()

    i = 0
    while result[i][1] != 0:
        u, v = result[i][0]
        # For directed graph, add edge from u to v or v to u based on larger influence
        if P_dict[(u, v)] > P_dict[(v, u)]:
            IG.add_edge(u, v)
        else:
            IG.add_edge(v, u)

        i += 1
        if i >= len(result):
            break

    return IG, result, A, P_dict
