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
from sampling.BNSampling import Mynew_forward_sampling
import datetime
from svd.MySvd import MY_SVD

Embedding_time = {}
Inference_time = {}
Embedding_Inference_time = {}


def M_choose_random_state(node, network):
    '''
    Assigns a random state to a given node, like assign 'small' to size
    @param node:
    @param network:
    @return:
    '''
    state_num_r = network['data'][node]['state_num']
    random_index = random.randint(0, state_num_r - 1)
    return random_index


# for node in nodes_sort:
#     print(' %s:%s ' %(node,M_choose_random_state(node, net)))


def choose_non_evidence_node(non_evidence_nodes):
    '''
    choose a random non-evidence node in the current iteration
    @param non_evidence_nodes:
    @return:
    '''
    return non_evidence_nodes[random.randint(0, len(non_evidence_nodes) - 1)]


def node_sim(node, network, my_svd, nodes_sim_store):
    node_state_combi = ['{node}_{state}'.format(node=node, state=state) for state in
                        range(network['data'][node]['state_num'])]
    parents_current = network['data'][node]['parents']
    childrens_current = network['data'][node]['children']
    pare_child = parents_current + childrens_current
    pare_child_state_combi = ['{pare_child}_{state}'.format(pare_child=node, state=state)
                              for node in pare_child
                              for state in range(network['data'][node]['state_num'])]

    for i in node_state_combi:
        node_index = n_row_index.index(i)  # acquire node index
        node_to_pare_child_sim_measure_not_null = {}  # record the not null the minimum Euclidean distance
        for j in pare_child_state_combi:
            pare_child_index = n_row_index.index(j)  # acquire parent_children index
            sim_measure = my_svd.gibbs_sim_measure(my_svd.ecludSim, node_index,
                                                   pare_child_index)  # ecludSim between embedded vectors
            #             print('pare_child_index:\n',pare_child_index)
            #             print('sim({i},{j}):{similarity}\n'.format(i=i,j=j,similarity=sim_measure))
            nodes_sim_store[node][i][j] = sim_measure  # store the similarity between node[i] and parent_children
            node_to_pare_child_sim_measure_not_null[
                (i, j)] = sim_measure  # store the similarity between node[i] and parent_children

    return nodes_sim_store

def M_test1_update_value(node, network, simulation, my_svd):
    '''
    update the value of node in previous iteration
    @param node:
    @param network:
    @param simulation:
    @param my_svd
    @return:
    '''

    node_state_combi = ['{node}_{state}'.format(node=node, state=state) for state in
                        range(network['data'][node]['state_num'])]
    parents_current = network['data'][node]['parents']
    childrens_current = network['data'][node]['children']
    pare_child_current = parents_current + childrens_current
    # simulation[-1]读取列表中第一个元素,simulation=[{'amenities':1, 'location':2, 'neighborhood':0}],初始化样本取值
    pare_child_current_state = ['{c_node}_{state}'.format(c_node=c_node, state=simulation[-1][c_node])
                                for c_node in pare_child_current]
    #     print('pare_child_current_state:',pare_child_current_state)
    node_to_pare_child_sim_min = {}  # the minimum Euclidean distance represents the max similarity
    sim_min_starttime = datetime.datetime.now()
    for i in node_state_combi:
        node_to_pare_child_sim_measure = {}
        node_to_pare_child_sim_measure_not_null = {}  # record the not null the minimum Euclidean distance
        for j in pare_child_current_state:
            sim_measure = nodes_sim_store[node][i][j]
            #             print('pare_child_index:\n',pare_child_index)
            #             print('sim({i},{j}):{similarity}\n'.format(i=i,j=j,similarity=sim_measure))
            node_to_pare_child_sim_measure[
                (i, j)] = sim_measure  # store the similarity between node[i] and parent_children
            if sim_measure != 0:
                node_to_pare_child_sim_measure_not_null[
                    (i, j)] = sim_measure  # store the similarity between node[i] and parent_children
        global sim_min
        sim_min = min(node_to_pare_child_sim_measure, key=lambda x: node_to_pare_child_sim_measure[x])
        # 保存nodei在不同状态下与其他节点的最小相似值
        node_to_pare_child_sim_min[sim_min] = node_to_pare_child_sim_measure[
            sim_min]  # store min similarity of differnt states of node[i]
    #         print('node_to_pare_child_sim_min:\n',node_to_pare_child_sim_min)
    sim_min_endtime = datetime.datetime.now()
    sim_min_time = sim_min_endtime - sim_min_starttime
    #     print('sim_min_time',sim_min_time)
    #     sim_min_reverse=[i for i in node_to_pare_child_sim_min.values() if i!=0]
    sim_min_reverse = []
    for i in node_to_pare_child_sim_min.values():
        if i != 0:
            sim_min_reverse.append(1 / i)
        else:
            sim_min_reverse.append(i)

    prob_norm_starttime = datetime.datetime.now()
    sim_min_sum = sum(sim_min_reverse)  # compute the sum of different node[i] min similarity
    #     print('sim_min_sum:\n',sim_min_sum,type(sim_min_sum))
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
    prob_norm_endtime = datetime.datetime.now()
    prob_norm_time = prob_norm_endtime - prob_norm_starttime
    #     print('prob_norm_time:',prob_norm_time)
    return index

def BNERS(network, evidence, node_to_query, niter):  # evidence is dict and node_to_query is dict
    simulation = []  # 样本集合
    #     nodes = nodes_sort
    d = {}  # d是起始样本D1
    for node in nodes:
        d[node] = M_choose_random_state(node, network)
    #     print('d:%s'%d)
    # Put evidence
    for node in evidence:
        d[node] = evidence[node]
    #     print('d:%s'%d)
    non_evidence_nodes = [node for node in nodes if node not in evidence.keys()]
    simulation.append(d.copy())  # 所有样本的集合列表
    #     print('initial_simulation:/n',simulation)
    #     print('initial_simulation:%s'%simulation)
    M_update_value_runtime = {}
    M_update_value_starttime = datetime.datetime.now()
    for count in range(niter - 1):
        # Pick up a random node to start
        current_node_to_update = choose_non_evidence_node(non_evidence_nodes)
        d[current_node_to_update] = M_test1_update_value(current_node_to_update, network, simulation, my_svd)
        simulation.append(d.copy())
    M_update_value_endtime = datetime.datetime.now()
    M_update_value_runtime['M_update_value_runtime'] = M_update_value_endtime - M_update_value_starttime
    print(M_update_value_runtime)
    to_query_node = list(node_to_query.keys())[0]
    count = {query_node_state: 0 for query_node_state in range(network['data'][to_query_node]['state_num'])}
    #     count = {val: 0 for val in network['data'][to_query_node]['state']}
    #     print('----------------------------------------')
    #     print('count',count)

    for i in range(len(simulation)):
        #         print(simulation[i])
        count[simulation[i][to_query_node]] += 1
    #     print('**********************************')
    print(count)
    # print(assignment[node_to_query])

    for l in count:
        probabilites[l] = count[l] / niter

    to_query_node_val = list(node_to_query.values())[0]
    probabilites_val = probabilites[to_query_node_val]
    return probabilites

convergence_one_evidence = {}
convergence_two_evidence = {}
convergence_three_evidence = {}

# 将{'location':'bad'}转化为{'location':1}的形式
def state_to_num(net, node):
    for keys, values in list(node.items()):
        nstate = net['data'][keys]['state']  # find node[i] state
        nstate_index = nstate.index(values)
        #         print('keys:',keys)
        node.update({keys: nstate_index})
    #         print('node:',node)
    return node

# 将probability={0:0.4,1:0.6}转化为{'old':0.4,'new':0.6}的形式
def pnum_to_state(net, node, probabilites):
    for keys, values in list(probabilites.items()):
        nstate = net['data'][list(node.keys())[0]]['state'][keys]
        probabilites.update({nstate: probabilites.pop(keys)})
        dict.pop
    return probabilites


if __name__ == '__main__':

    reader = BIFReader(os.getenv('SHARED_DIR') + '/Builds/qizhiwei/network.bif')  # network
    # reader = BIFReader('data/dataset/alarm.bif')  #Alarm
    # reader = BIFReader('data/dataset/hepar2.bif')   #Large BN
    # reader = BIFReader('data/dataset/andes.bif')  #Very large BN
    network = reader.get_model()

    G = Digraph('network')
    nodes = network.nodes()
    print('nodes_num:', len(nodes))
    nodes_sort = list(nx.topological_sort(network))
    print('nodes_sort:', nodes_sort)

    edges = network.edges()
    print('\nedges_num:', len(edges))
    for a, b in edges:
        G.edge(a, b)
    var_card = {node: network.get_cardinality(node) for node in nodes_sort}
    print('var_card:', var_card)
    net = ConstructParentsChildrenBN(reader, network)

    Sampling_time = {}
    Sampling_starttime = datetime.datetime.now()

    # inference = BayesianModelSampling(student)
    # samples=inference.forward_sample(size=100, return_type='dataframe')  # return_type='recarray/dataframe'
    # print(samples)
    size = 8000
    samples_Mynew = Mynew_forward_sampling(network).My_forward_sample(size, return_type='dataframe')
    # print('samples_Mynewo:\n',samples_Mynew,'\n')
    # print(samples_Mynew.values,len(samples_Mynew))
    # student.get_cardinality('diff')
    Sampling_endtime = datetime.datetime.now()
    Sampling_time['Sampling_time'] = (Sampling_endtime - Sampling_starttime).seconds
    print(Sampling_time)

    Generation_PMI_time = {}
    Generation_PMI_starttime = datetime.datetime.now()
    ##initialize E_ijk matrix
    # nodes of graph 's state are counted initially
    nodes_state_count = {node: {node_state: 0 for node_state in range(network.get_cardinality(node))} for node in nodes}
    # print('nodes_state_count:\n',nodes_state_count)

    # nodes of graph 's state are counted initially

    edges_state_count = {edge:
                             {state_combination: 0 for state_combination in
                              itertools.product(*[range(network.get_cardinality(node))
                                                  for node in edge])}
                         for edge in edges}
    edges_state_values = copy.deepcopy(edges_state_count)  # copy.deepcopy
    # print('edges_state_count:\n',edges_state_count)
    # print('edges_state_values:\n',edges_state_values)
    # print(100*'-'+'\n')

    # edges_state matrix is initialized
    e_row_index = edges
    e_column_index = set()
    for edge in edges:
        for state_combination in itertools.product(*[range(network.get_cardinality(node)) for node in edge]):
            e_column_index.add(state_combination)
    e_column_index = list(e_column_index)
    e_column_index.sort()
    # print('e_row_index:',e_row_index)
    # print('e_column_index:',e_column_index)
    E_ijk = pd.DataFrame(0, index=e_row_index, columns=e_column_index)
    # print(E_ijk)

    # nodes of graph 's state are counted, calculate #(X_i)和#(X_j)
    for node in nodes:
        for i in range(len(samples_Mynew[node])):
            nodes_state_count[node][samples_Mynew[node][i]] += 1
    # print('nodes_state_count:\n',nodes_state_count)

    E_ijk_time = {}
    E_ijk_starttime = datetime.datetime.now()
    # edges of graph 's state are counted， and calculate #(X_i,X_j)*|D|/#(X_i)#(X_j)
    for edge in edges:
        for state_combination in itertools.product(*[range(network.get_cardinality(node)) for node in edge]):
            #         print('edge：{edge} state:{state_combination}'.format(edge=edge,state_combination=state_combination))
            for i in range(len(samples_Mynew)):
                if samples_Mynew.loc[i, edge[0]] == state_combination[0] and samples_Mynew.loc[i, edge[1]] == \
                        state_combination[1]:
                    edges_state_count[edge][state_combination] += 1
            # calculate #(X_i,X_j)*|D|/#(X_i)#(X_j)
            edges_state_values[edge][state_combination] = round((edges_state_count[edge][state_combination] * size)
                                                                / (nodes_state_count[edge[0]][state_combination[0]]
                                                                   * nodes_state_count[edge[0]][state_combination[0]]),
                                                                3)
            E_ijk.loc[edge, state_combination] = edges_state_values[edge][state_combination]
            #             print(samples_Mynew.loc[i,edge[0]],samples_Mynew.loc[i,edge[1]])
    E_ijk_endtime = datetime.datetime.now()
    E_ijk_time['E_ijk_time'] = (E_ijk_endtime - E_ijk_starttime).seconds
    print(E_ijk_time)
    # print('edges_state_count:\n',edges_state_count)
    # print('edges_state_values:\n',edges_state_values)
    # print('E_ijk:\n',E_ijk)

    # generate the index of node, suach as: [X_1,X_2,X_3,...,X_n]
    n_row_index = []
    for i in range(len(nodes_sort)):
        for j in range(var_card[nodes_sort[i]]):
            n_row_index.append('{u}_{s}'.format(u=nodes_sort[i], s=j))
            # print('\n',n_row_index)

    # initialize node_state matirx
    N_ijk = pd.DataFrame(0, index=n_row_index, columns=n_row_index)
    # print(N_ijk)
    PMI_ijk = pd.DataFrame(0, index=n_row_index, columns=n_row_index)
    # print('PMI_ijk:\n',PMI_ijk)

    # construct PMI matirx logN_ijk-logk
    # first construct the nodes_state matrix, make the edges_state matrix transformed to nodes_state matrix
    k = 1
    for edge in edges:
        for state_combination in itertools.product(*[range(network.get_cardinality(node)) for node in edge]):
            #         print('state_combination:',state_combination)
            edge_0 = '{u}_{s}'.format(u=edge[0], s=state_combination[0])
            edge_1 = '{u}_{s}'.format(u=edge[1], s=state_combination[1])
            #         print(edge_0,edge_1)
            N_ijk.loc[edge_0, edge_1] = edges_state_values[edge][state_combination]
            # PMI=max(PMI,0)
            if N_ijk.loc[edge_0, edge_1] == 0:
                PMI_ijk.loc[edge_0, edge_1] = 0
            else:
                PMI_ijk.loc[edge_0, edge_1] = round(np.log(N_ijk.loc[edge_0, edge_1]) - np.log(k), 3)
                if PMI_ijk.loc[edge_0, edge_1] < 0:
                    PMI_ijk.loc[edge_0, edge_1] = 0
    # print('N_ijk:\n',N_ijk.values)
    # print('PMI_ijk',PMI_ijk.values)
    Generation_PMI_endtime = datetime.datetime.now()
    Generation_PMI_time['Generation_PMI_time'] = (Generation_PMI_endtime - Generation_PMI_starttime).seconds
    print(Generation_PMI_time)

    print('降维前的维度:', len(n_row_index))

    Embedding_time = {}
    Embed_starttime = datetime.datetime.now()
    # BN embedding


    data = PMI_ijk.values
    '''按照前k个奇异值的平方和占总奇异值的平方和的百分比percentage来确定k的值,
        后续计算SVD时需要将原始矩阵转换到k维空间'''
    my_svd = MY_SVD(data, 0.99)
    newData, VT_k = my_svd.DimReduce_k()

    nodes_sim_store = {node: {'{node}_{state}'.format(node=node, state=state): {}
                              for state in range(net['data'][node]['state_num'])}
                       for node in nodes}
    # print(nodes_sim_store)
    for node in nodes:
        one_node_sim_store = node_sim(node, net, my_svd, nodes_sim_store)


    niter = 10000
    node_to_query = {'location': 'good'}
    evidence = {'amenities': 'lots'}  # 'neighborhood':'bad','age':'old'} #'neighborhood':'bad','age':'old'

    # node_to_query = {'SNode_3':'false'}  #'Steatosis':'present', hepatotoxic, transfusion, injections, obesity, flatulence
    # evidence={'SNode_4':'false'}#'surgery':'present', diabetes
    # # evidence={'PBC':'present','gallstones':'present'} #'hospital':'present','surgery':'present','gallstones':'present',
    # #'hospital':'present','surgery':'present','gallstones':'present','choledocholithotomy':'present'

    node_to_query = state_to_num(net, node_to_query)
    # print('node_to_query:\n',node_to_query)
    # nq_index = node_query_state.index(list(node_to_query.values())[0])
    # print('nq_index:',nq_index)

    evidence = state_to_num(net, evidence)
    # print('evidence:',evidence)

    # probabilites={0: 1.0, 1: 0.0, 2: 0.0}
    # probabilites = pnum_to_state(net,node_to_query,probabilites)
    # print('probabilites:',probabilites)

    Inference_starttime = datetime.datetime.now()
    probabilites = {}
    # num_update_list = np.arange(1, 12, 2)
    # print('evidence:%s'%evidence)
    # for iterations in num_update_list:

    probabilites = BNERS(net, evidence, node_to_query, niter)
    # print('probabilites:%s'%probabilites)
    # probabilites = pnum_to_state(net,node_to_query,probabilites)
    # print('probabilites:%s'%probabilites)
    print("P({query}|{evid})={prob}".format(query=list(node_to_query.keys()), evid=evidence, prob=probabilites))
    Inference_endtime = datetime.datetime.now()
    Inference_time = Inference_endtime - Inference_starttime  # Alarm、HEPAR2、ANDES
    print('Inference_time:', Inference_time)
    # convergence_one_evidence[niter]=round(probabilites['present'],3)
    # print(convergence_one_evidence)
