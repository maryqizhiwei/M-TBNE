import os
import torch
import dgl
import pandas as pd
import numpy as np
import itertools
import networkx as nx
from typing import Optional
from graphviz import Digraph
from dgl.data import QM9
from pgmpy.readwrite import BIFReader
from torch_geometric.data import Data, Dataset
from pgmpy.sampling import BayesianModelSampling
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from graphormer.data import register_dataset
import torch.distributed as dist
import pickle

@register_dataset("my_dataset")

def create_customized_dataset():
    reader = BIFReader('dataset copy/child.bif')
    network = reader.get_model()

    G = Digraph('network') 
    nodes = network.nodes()
    nodes_sort = list(nx.topological_sort(network))

    edges = network.edges()
    print('\nedges_num:', len(edges))

    for a, b in edges:
        G.edge(a, b)
    var_card = {node: network.get_cardinality(node) for node in nodes_sort}
    print('var_card:', var_card)
    size =10000

    def forward_sample_bayesian_network(model, size):
        infer = BayesianModelSampling(model)
        topological_order = infer.topological_order

        types = [(var_name, 'int') for var_name in topological_order]
        sampled = np.zeros(size, dtype=types).view(np.recarray)

        for node in topological_order:
            cpd = model.get_cpds(node)
            states = range(infer.cardinality[node])
            evidence = cpd.variables[:0:-1]

            if evidence:
                evidence_values = np.vstack([sampled[i] for i in evidence])
                cached_values = pre_compute_reduce(model, node, infer.cardinality)
                weights = [cached_values[tuple(e)] for e in evidence_values.T]

            else:
                weights = cpd.values
            for i in range(size):
                if evidence:
                    p = weights[i]
                else:
                    p = weights
                sampled[node][i] = np.random.choice(states, p=p)

        return sampled

    def pre_compute_reduce(model, variable, cardinality):
        variable_cpd = model.get_cpds(variable)
        variable_evid = variable_cpd.variables[:0:-1]
        cached_values = {}

        for state_combination in itertools.product(*[range(cardinality[var]) for var in variable_evid]):
            states = list(zip(variable_evid, state_combination))
            cached_values[state_combination] = variable_cpd.reduce(states, inplace=False).values
        return cached_values

    def sampled_node_features(model, sampled_data):
        all_sample_features = []
        infer = BayesianModelSampling(model)
        topological_order = infer.topological_order

        max_length = 0
        for node in topological_order:
            cpd = model.get_cpds(node)
            cpd_vector = cpd.values.flatten()
            num_states = cpd.variable_card
            state_vectors = np.array_split(cpd_vector, num_states)
            max_length = max(max_length, len(state_vectors[0]))
        print(max_length)
        for sample in sampled_data:
            sample_features = []
            for node in topological_order:
                cpd = model.get_cpds(node)
                cpd_vector = cpd.values.flatten()
                #print('cpd_vector:', cpd_vector)
                num_states = cpd.variable_card
                state_vectors = np.array_split(cpd_vector, num_states)
                #print('state_vectors:', state_vectors)

                node_features = [np.zeros(max_length) for _ in range(num_states)]
                
                state = sample[node]
                state_vector = state_vectors[state]
                #print('state_vector:', state_vector)
                if len(state_vector) < max_length:
                    state_vector = np.pad(state_vector, (0, max_length - len(state_vector)))
                node_features[state] = state_vector * 100
                sample_features.extend(node_features)
            #print('sample_features:', sample_features)
            # 将状态向量重塑为一个行向量
            sample_features = np.vstack(sample_features)
            #print('sample_features:', sample_features)
            all_sample_features.append(sample_features)
        return all_sample_features

    sampled_data = forward_sample_bayesian_network(network, size)
    sample_features = sampled_node_features(network, sampled_data)
    #print('sample_features:\n', sample_features[1])
    nodes_state_count = {node: {node_state: 0 for node_state in range(network.get_cardinality(node))} for node in nodes_sort}
    edges_state_count = {edge:
                                {state_combination: 0 for state_combination in
                                itertools.product(*[range(network.get_cardinality(node))
                                                    for node in edge])}
                            for edge in edges}
    edges_state_values = {edge: {state_combination: 0 for state_combination in itertools.product(*[range(network.get_cardinality(node)) for node in edge])} for edge in edges}
    e_row_index = edges
    e_column_index = set()
    for edge in edges:
        for state_combination in itertools.product(*[range(network.get_cardinality(node)) for node in edge]):
            e_column_index.add(state_combination)
    e_column_index = list(e_column_index)
    e_column_index.sort()
    E_ijk = pd.DataFrame(0, index=e_row_index, columns=e_column_index)
    E_ijk.columns = pd.MultiIndex.from_tuples(E_ijk.columns)
    E_ijk = E_ijk.astype(float)
    #print('E_ijk:\n', E_ijk)
    for node in nodes_sort:
        for i in range(len(sampled_data[node])):
            nodes_state_count[node][sampled_data[node][i]] += 1
    sampled_data = pd.DataFrame(sampled_data)        
    #print('sampled_data:\n', sampled_data)

    prior_probabilities = {}
    for node, state_counts in nodes_state_count.items():
        total_counts = sum(state_counts.values())
        prior_probabilities[node] = {state: count / total_counts for state, count in state_counts.items()}

    prior_probabilities_vector = []
    for node_probs in prior_probabilities.values():
        for prob in node_probs.values():
            prior_probabilities_vector.append(prob)
    print(prior_probabilities_vector)

    for edge in edges:
        for state_combination in itertools.product(*[range(network.get_cardinality(node)) for node in edge]):
            matching_rows = sampled_data[(sampled_data[edge[0]] == state_combination[0]) & (sampled_data[edge[1]] == state_combination[1])]
            count = matching_rows.shape[0]
            edges_state_count[edge][state_combination] += count

            denom = (nodes_state_count[edge[0]][state_combination[0]] * nodes_state_count[edge[1]][state_combination[1]])
            if denom != 0:
                edges_state_values[edge][state_combination] = round((count * size) / denom, 3)
                E_ijk.loc[edge, state_combination] = edges_state_values[edge][state_combination]

    n_row_index = []
    for i in range(len(nodes_sort)):
        for j in range(var_card[nodes_sort[i]]):
            n_row_index.append('{u}_{s}'.format(u=nodes_sort[i], s=j))
    #print('n_row_index:', n_row_index)
    N_ijk = pd.DataFrame(0, index=n_row_index, columns=n_row_index)
    N_ijk = N_ijk.astype(float)
    PMI_ijk = pd.DataFrame(0, index=n_row_index, columns=n_row_index)
    PMI_ijk = PMI_ijk.astype(float)

    k = 1
    for edge in edges:
        for state_combination in itertools.product(*[range(network.get_cardinality(node)) for node in edge]):
            edge_0 = '{u}_{s}'.format(u=edge[0], s=state_combination[0])
            edge_1 = '{u}_{s}'.format(u=edge[1], s=state_combination[1])
            N_ijk.loc[edge_0, edge_1] = edges_state_values[edge][state_combination]
            if N_ijk.loc[edge_0, edge_1] == 0:
                PMI_ijk.loc[edge_0, edge_1] = 0
            else:
                PMI_ijk.loc[edge_0, edge_1] = round(10 * edges_state_count[edge][state_combination] / size * np.log(N_ijk.loc[edge_0, edge_1]) , 3)
                if PMI_ijk.loc[edge_0, edge_1] < 0:
                    PMI_ijk.loc[edge_0, edge_1] = 0
    #print(PMI_ijk)
    PMI = []
    adjacency_matrices = []
    for idx, row in sampled_data.iterrows():
        #print('row:', row)
        adjacency_matrix = pd.DataFrame(0, index=n_row_index, columns=n_row_index)
        adjacency_matrix = adjacency_matrix.astype(float)
        for edge in edges:
            for state_combination in itertools.product(*[range(network.get_cardinality(node)) for node in edge]):
                edge_0 = '{u}_{s}'.format(u=edge[0], s=state_combination[0])
                edge_1 = '{u}_{s}'.format(u=edge[1], s=state_combination[1])
                if row[edge[0]] == state_combination[0] and row[edge[1]] == state_combination[1]:
                    adjacency_matrix.loc[edge_0, edge_1] = 1
        adjacency_matrices.append(adjacency_matrix)
        PMI.append(PMI_ijk * 100)
        #print('adjacency_matrix:\n', adjacency_matrix)
    #print('adjacency_matrices:\n', adjacency_matrices)

    class MakePyGDataset(Dataset):
        def __init__(self, adjacency_matrices, PMI, sample_features, sampled_data, prior_probabilities_vector, transform=None, pre_transform=None):
            self.adjacency_matrices = adjacency_matrices
            self.PMI = PMI
            self.sample_features = sample_features
            self.sampled_data = sampled_data
            self.prior_probabilities_vector = prior_probabilities_vector
            super(MakePyGDataset, self).__init__('.', transform, pre_transform)

        def len(self):
            return len(self.adjacency_matrices)

        def get(self, idx):
            adj_matrix = self.adjacency_matrices[idx].values
            pmi_matrix = self.PMI[idx].values  # Assuming DataFrame
            sample_features_matrix = self.sample_features[idx]
            # 计算每个节点的状态的所有条件概率
            sample_features_matrix = sample_features_matrix[:, :12]
            
            x = torch.tensor(sample_features_matrix, dtype=torch.long)  # Your node features
            rows, cols = np.nonzero(adj_matrix)
            edges = np.array([rows, cols])  # 先将列表转换为一个 numpy 数组
            edges = torch.tensor(edges, dtype=torch.long)  # 然后将 numpy 数组转换为一个 PyTorch 张量
            edge_index = edges
            edge_values = pmi_matrix[rows, cols]
            # 将 pmi_matrix 中的值赋给 edge_attr
            edge_attr = torch.tensor(edge_values, dtype=torch.long)

            # 添加反向边
            reverse_edges = np.array([cols, rows])  # 反向边的起点和终点与原边相反
            reverse_edges = torch.tensor(reverse_edges, dtype=torch.long)
            reverse_edge_attr = torch.tensor(edge_values, dtype=torch.long)  # 反向边的属性与原边相同

            edge_index = torch.cat([edges, reverse_edges], dim=1)
            edge_attr = torch.cat([edge_attr, reverse_edge_attr])
            
            '''
            # e.g., y (target), edge_attr, etc.
            #y = torch.tensor([0], dtype=torch.long)
            y = torch.tensor(0, dtype=torch.float32)
            #y = torch.tensor(self.prior_probabilities_vector, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            '''
            

            
            # 随机遮掩节点特征
            num_nodes = len(sample_features_matrix)
            num_nodes_to_mask = int(0.1 * num_nodes)  # 例如，遮掩10%的节点
            nodes_to_mask = np.random.choice(num_nodes, num_nodes_to_mask, replace=False)

            for node_id in nodes_to_mask:
                x[node_id] = torch.zeros_like(x[node_id])
            #x = torch.zeros_like(x)
            # 创建一个表示节点存在与否的向量
            node_existence1 = torch.zeros(num_nodes, dtype=torch.float)
            for node_id in edge_index[0]:
                node_existence1[node_id] = 1

            
            data = Data(x=x, edge_index=edge_index, y=node_existence1, edge_attr=edge_attr)
            
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            return data

    dataset = MakePyGDataset(adjacency_matrices, PMI, sample_features, sampled_data, prior_probabilities_vector)


    #print(dataset)
    #print(dataset[0])
    #print(dataset[0].y)
    #print(dataset[0].x)
    #print(dataset[0].edge_attr)
    #print(dataset[0].edge_index)

    print("import successful")

    num_graphs = len(dataset)

    # customized dataset split
    train_valid_idx, test_idx = train_test_split(
        np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=num_graphs // 5, random_state=0
    )
    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg"
    }