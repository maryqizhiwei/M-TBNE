import os
import torch
import dgl
import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional
from graphviz import Digraph
from dgl.data import QM9
from pgmpy.readwrite import BIFReader
from torch_geometric.data import Data, Dataset
from pgmpy.sampling import BayesianModelSampling
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
import torch.distributed as dist
from scipy.spatial import distance
from scipy.special import softmax
from pgmpy.models import BayesianModel
import networkx as nx
import itertools
import os
from matplotlib import rc
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pgmpy.readwrite import BIFReader
from graphviz import Digraph
from BN.Construct_BN import ConstructParentsChildrenBN
from pgmpy.models.BayesianModel import BayesianModel
import copy
import pandas as pd
#from sampling.BNSampling import Mynew_forward_sampling
import datetime
from svd.MySvd import MY_SVD
import time
# 读取 JSON 文件
def read_infer_json(path):
    infer_json = pd.read_json(path)
    infer_json = infer_json.T
    return infer_json

infer = read_infer_json('../实验部分/infer43.json')
df = pd.read_csv('sampled_data.csv')
#print(df)
# 使用 BIFReader 读取 BIF 文件并获取模型
reader = BIFReader('dataset/win95pts.bif')
network = reader.get_model()

G = nx.DiGraph()

global nodes_sort
# 获取节点和边
nodes = network.nodes()
nodes_sort = list(nx.topological_sort(network))
edges = network.edges()


for edge in edges:
    G.add_edge(*edge)
var_card = {node: network.get_cardinality(node) for node in nodes_sort}

n_row_index = []
for i in range(len(nodes_sort)):
    for j in range(var_card[nodes_sort[i]]):
        n_row_index.append('{u}_{s}'.format(u=nodes_sort[i], s=j))

infer.columns = n_row_index

global nodes_sim_store

net = ConstructParentsChildrenBN(reader, network) #'V', 'E', 'data': 'BirthAsphyxia':'state', 'state_num', 'parents','children','cpds'(父母状态*当前状态) 
#print(net)
nodes_sim_store = {node: {'{node}_{state}'.format(node=node, state=state): {}
                            for state in range(net['data'][node]['state_num'])}
                    for node in nodes} #'BirthAsphyxia' ：'BirthAsphyxia_0'：{}

def node_sim(node, network, my_svd, nodes_sim_store, infer): #net = network
    node_state_combi = ['{node}_{state}'.format(node=node, state=state) for state in
                        range(network['data'][node]['state_num'])] #包含0的状态组合
    parents_current = network['data'][node]['parents']
    childrens_current = network['data'][node]['children']
    pare_child = parents_current + childrens_current
    pare_child_state_combi = ['{pare_child}_{state}'.format(pare_child=node, state=state)
                              for node in pare_child
                              for state in range(network['data'][node]['state_num'])]

    for i in node_state_combi:
        node_index = n_row_index.index(i)  # acquire node index
        node_to_pare_child_sim_measure_not_null = {}  # record the not null the minimum Euclidean distance
        for j in pare_child_state_combi: #对当前节点状态的父母节点进行遍历
            pare_child_index = n_row_index.index(j)  # 得到60行中的索引
            sim_measure = my_svd.gibbs_sim_measure(my_svd.cosSim, node_index,
                                                   pare_child_index, infer)  # ecludSim between embedded vectors
            #             print('pare_child_index:\n',pare_child_index)
            #             print('sim({i},{j}):{similarity}\n'.format(i=i,j=j,similarity=sim_measure))
            nodes_sim_store[node][i][j] = sim_measure
            #print('sim({i},{j}):{similarity}\n'.format(i=i, j=j, similarity=sim_measure))
            node_to_pare_child_sim_measure_not_null[(i, j)] = sim_measure  # store the similarity between node[i] and parent_children

    return nodes_sim_store

my_svd = MY_SVD()

for node in nodes:
    one_node_sim_store = node_sim(node, net, my_svd, nodes_sim_store, infer)

def state_to_num(net, node):
    for keys, values in list(node.items()):
        nstate = net['data'][keys]['state']  # find node[i] state
        nstate_index = nstate.index(values)
        #         print('keys:',keys)
        node.update({keys: nstate_index})
    #         print('node:',node)
    return node


def M_choose_random_state(node, network):
    state_num_r = network['data'][node]['state_num']
    random_index = random.randint(0, state_num_r - 1)
    return random_index

def choose_non_evidence_node(non_evidence_nodes):
    return non_evidence_nodes[random.randint(0, len(non_evidence_nodes) - 1)]

def M_test1_update_value(node, network, simulation, my_svd): #current_node_to_update ，net，simulation.append(d.copy())  # 所有样本的集合列表
    node_state_combi = ['{node}_{state}'.format(node=node, state=state) for state in
                        range(network['data'][node]['state_num'])]
    parents_current = network['data'][node]['parents']
    childrens_current = network['data'][node]['children']
    pare_child_current = parents_current + childrens_current
    # simulation[-1]读取列表中第一个元素,simulation=[{'amenities':1, 'location':2, 'neighborhood':0}],初始化样本取值
    pare_child_current_state = ['{c_node}_{state}'.format(c_node=c_node, state=simulation[-1][c_node]) #父母节点和子节点的状态固定
                                for c_node in pare_child_current]
    #     print('pare_child_current_state:',pare_child_current_state)
    node_to_pare_child_sim_min = {}  # the minimum Euclidean distance represents the max similarity
    for i in node_state_combi: #对当前节点的状态进行遍历
        node_to_pare_child_sim_measure = {}
        node_to_pare_child_sim_measure_not_null = {}  # record the not null the minimum Euclidean distance
        for j in pare_child_current_state: #对当前节点状态的父母孩子节点进行遍历
            sim_measure = nodes_sim_store[node][i][j]
            node_to_pare_child_sim_measure[
                (i, j)] = sim_measure  # store the similarity between node[i] and parent_children
            if sim_measure != 0:
                node_to_pare_child_sim_measure_not_null[
                    (i, j)] = sim_measure  # store the similarity between node[i] and parent_children
        global sim_min
        sim_min = min(node_to_pare_child_sim_measure, key=lambda x: node_to_pare_child_sim_measure[x])
        # 保存nodei在不同状态下与其他节点的最小相似值
        node_to_pare_child_sim_min[sim_min] = node_to_pare_child_sim_measure[sim_min]  # store min similarity of differnt states of node[i]
    sim_min_reverse = []
    for i in node_to_pare_child_sim_min.values():
        if i != 0:
            sim_min_reverse.append(1 / i)
        else:
            sim_min_reverse.append(i)

    sim_min_sum = sum(sim_min_reverse)  # compute the sum of different node[i] min similarity
    # 当节点嵌入向量之间的相似性之和不为0时，执行以下基于权重的取值
    if sim_min_sum != 0:
        prob_norm = np.array(sim_min_reverse) / sim_min_sum
    else:
        if parents_current != []:
            values_parents = [network['data'][parent]['state'][simulation[-1][parent]] for parent in parents_current]
            prob_norm = network['data'][node]['cpds'][str(values_parents)]
        else:
            prob_norm = [i[0] for i in net['data'][node]['cpds'].values()]
    #     print('prob_norm:\n',prob_norm)
    prob_norm = np.cumsum(prob_norm)
    #     print('prob_norm:\n',prob_norm)
    choice = random.random()
    #     print('random_choice:',choice)
    index = np.argmax(prob_norm > choice)  # make the max similarity more easily to generate and 取元素最大值所对应的索引

    return index



def Inference(network, evidence, node_to_query, niter, df):  # evidence is dict and node_to_query is dict
    simulation = []  # 样本集合
    #     nodes = nodes_sort
    d = {}  # d是起始样本D1
    for node in nodes_sort:
        d[node] = M_choose_random_state(node, network)
    for node in evidence:
        d[node] = evidence[node]
    non_evidence_nodes = [node for node in nodes if node not in evidence.keys()]
    simulation.append(d.copy())  # 所有样本的集合列表
    for count in range(niter - 1):
        # Pick up a random node to start
        current_node_to_update = choose_non_evidence_node(non_evidence_nodes)
        d[current_node_to_update] = M_test1_update_value(current_node_to_update, network, simulation, my_svd)
        simulation.append(d.copy())

    to_query_node = list(node_to_query.keys())[0]
    count = {query_node_state: 0 for query_node_state in range(network['data'][to_query_node]['state_num'])}
    #     count = {val: 0 for val in network['data'][to_query_node]['state']}
    #     print('----------------------------------------')
    #     print('count',count)

    for i in range(len(simulation)):
        #         print(simulation[i])
        count[simulation[i][to_query_node]] += 1
    #     print('**********************************')
    #print(count)
    # print(assignment[node_to_query])

    for l in count:
        probabilites[l] = count[l] / niter
        
    to_query_node_val = list(node_to_query.values())[0]
    probabilites_val = probabilites[to_query_node_val]

    return probabilites_val


niter = 10000

node_to_query = {'CmpltPgPrntd': 'Yes'} #Slow
evidence = {'PrtMem': 'Less_than_2Mb'}
  # 'neighborhood':'bad','age':'old'} #'neighborhood':'bad','age':'old'

node_to_query = state_to_num(net, node_to_query)
evidence = state_to_num(net, evidence)


#print('probabilites_forward:', probabilites_forward)

print(node_to_query)
print(evidence)


probabilites = {}
start_time = time.time()
probabilites = Inference(net, evidence, node_to_query, niter, df)
end_time = time.time()
print('time:', end_time - start_time)
query_and_evidence = {**node_to_query, **evidence}
grouped_A = df.groupby(list(query_and_evidence.keys())).size()
#print(grouped_A)
count_A = grouped_A.loc[tuple(query_and_evidence.values())]
grouped_B = df.groupby(list(evidence.keys())).size()
count_B = grouped_B.loc[tuple(evidence.values())]
probabilites_f= count_A / count_B
print("probabilites_f", probabilites_f)
#probabilites_val = (probabilites[to_query_node_val]  + probabilites_f)/2

print("P({query}|{evid})={prob}".format(query=node_to_query, evid=evidence, prob=probabilites))


