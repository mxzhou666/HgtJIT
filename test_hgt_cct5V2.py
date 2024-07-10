import os
import sys
import time

import dgl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from hgt_model import HGTClassfication, HGTTest, HGTTrain, GraphDataset
from PGCN_noAST import PGCN, PGCNTest, PGCNTrain
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from models import CodeChangeModel, build_load_model
from transformers import RobertaTokenizer

from preproc import extract_graphs, construct_graphs
from collections import defaultdict
import random
import concurrent.futures
from functools import partial

logsPath = './logs_hgt/'
testPath = './testdata/'                 # PDG
# testPath = './dataset/testdata_ast/'   # CPG
mdlsPath = './models/'

# parameters
_CLANG_  = 1
_NETXARCHT_ = 'HGT'
_BATCHSIZE_ = 64
_epochs_ = 50
# dim_features = 768
start_time = time.time() #mark start time

class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def RunTime():
    pTime = ' [TIME: ' + str(round((time.time() - start_time), 2)) + ' sec]'
    return pTime

count_extruct_and_constructgraphs = 0
def extract_construct_single_graph(filename, cct5_model, tokenizer):
    global count_extruct_and_constructgraphs
    count_extruct_and_constructgraphs += 1
    print(f"Processed {filename} (Total: {count_extruct_and_constructgraphs} files)")
    nodes, edges, nodes0, edges0, nodes1, edges1, label = extract_graphs.ReadFile(filename)
    nodeDict, edgeIndex, edgeAttr = construct_graphs.ProcEdges(edges)
    nodeAttr, nodeType, nodeInvalid = construct_graphs.ProcNodes(nodes, nodeDict, cct5_model, tokenizer)


    label = [int(label)]
    savename = filename + '_cct5v2.npz'

    np.savez(savename, edgeIndex=edgeIndex, edgeAttr=edgeAttr, nodeAttr=nodeAttr, nodeType=nodeType, label=label,
             nodeDict=nodeDict)

    return f'[INFO] <main> save the graph information into numpy file: [' + savename + '] '

# CCT5 to embedding
def extruct_and_constructgraphs():
    cct5_model, tokenizer = build_load_model()
    count_extruct_and_constructgraphs = 0
    filenames = []
    constrcut_fixed_time = "2023-09-09 18:00:00"
    for root, ds, fs in os.walk(testPath):
        if 'out_slim_ninf_noast_n1_w.log_cct5v2.npz' in fs:
            filename = os.path.join(root, 'out_slim_ninf_noast_n1_w.log_cct5v2.npz').replace('\\', '/')
            modified_time = os.path.getmtime(filename)
            modified_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_time))
            if modified_time > constrcut_fixed_time:
                count_extruct_and_constructgraphs += 1
                continue
        for file in fs:
            if ('.log' == file[-4:]):
                filename = os.path.join(root, file).replace('\\', '/')
                filenames.append(filename)
                extract_construct_single_graph(filename, cct5_model, tokenizer)
    # partial_func = partial(extract_construct_single_graph, cct5_model=cct5_model,
    #                        tokenizer=tokenizer)
    #
    # with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    #     results = executor.map(partial_func, filenames)
    #
    # for result in results:
    #     print(result)

# get dataset
def GetDataset(path=None):
    # check.
    if None == path:
        print('[Error] <GetDataset> The method is missing an argument \'path\'!')
        return [], []
    print('start load data!')
    files = []
    graph_list = []
    labels = []
    for root, _, filelist in os.walk(path):
        # if len(labels) >= 20:
        #     break
        # if root != './testdata/787e1353917a99982daa6c277b623c366d671a3e':
        #     continue
        # print('load data:' + str(len(labels)))
        for file in filelist:
            # if file[-7:] == '_np.npz':
            if file[-11:] == '_word52.npz':
            # if file[-11:] == '_cct5v2.npz':
            # if file[-17:] == 'n1_w.log_cct5.npz':
                # read a numpy graph file.
                graph = np.load(os.path.join(root, file), allow_pickle=True)
                files.append(root.split('/')[-1])

                # 异构图Transformer
                edgeIndex = graph['edgeIndex']
                src_node = edgeIndex[0]
                dst_node = edgeIndex[1]
                nodeAttr = graph['nodeAttr']
                edgeAttr = graph['edgeAttr']
                # 前两位代表变更前后10是删除，00是不变，01是新增
                # 后3位代表边的类型，100CDG  010DDG  001AST
                nodeType  = graph['nodeType']
                # print(nodeAttr.shape)
                # -1为变更前， 0为不变， 1为变更后
                nodeDict = {}
                keep_node = []
                del_node = []
                add_node = []
                keep_index = 0
                add_index = 0
                del_index = 0
                # 按类型分配新的索引，dgl构建异构图，节点ID是按类型分类后的新的id
                for id, type in enumerate(nodeType):
                    if type == '0':
                        nodeDict[id] = keep_index
                        # keep_node.append(nodeAttr[id])
                        keep_node.append(torch.tensor(nodeAttr[id]))
                        keep_index += 1
                    elif type == '1':
                        nodeDict[id] = add_index
                        # add_node.append(nodeAttr[id])
                        add_node.append(torch.tensor(nodeAttr[id]))
                        add_index += 1
                    elif type == '-1':
                        nodeDict[id] = del_index
                        # del_node.append(nodeAttr[id])
                        del_node.append(torch.tensor(nodeAttr[id]))
                        del_index += 1
                typeDict = {'-1': 'del', '0': 'keep', '1': 'add'}
                edges = defaultdict(lambda: ([], []))
                edge_no = 0
                for edge in edgeAttr:
                    if np.array_equal(edge[-3:], np.array([1, 0, 0])):
                        src = src_node[edge_no]
                        dst = dst_node[edge_no]
                        src_type = typeDict[nodeType[src]]
                        dst_type = typeDict[nodeType[dst]]
                        edges[(src_type, 'CDG', dst_type)][0].append(nodeDict[src])
                        edges[(src_type, 'CDG', dst_type)][1].append(nodeDict[dst])
                    elif np.array_equal(edge[-3:], np.array([0, 1, 0])):
                        src = src_node[edge_no]
                        dst = dst_node[edge_no]
                        src_type = typeDict[nodeType[src]]
                        dst_type = typeDict[nodeType[dst]]
                        edges[(src_type, 'DDG', dst_type)][0].append(nodeDict[src])
                        edges[(src_type, 'DDG', dst_type)][1].append(nodeDict[dst])
                    elif np.array_equal(edge[-3:], np.array([0, 0, 1])):
                        src = src_node[edge_no]
                        dst = dst_node[edge_no]
                        src_type = typeDict[nodeType[src]]
                        dst_type = typeDict[nodeType[dst]]
                        edges[(src_type, 'AST', dst_type)][0].append(nodeDict[src])
                        edges[(src_type, 'AST', dst_type)][1].append(nodeDict[dst])
                    else:
                        print(111)
                    edge_no = edge_no + 1

                for edge_type_str, (src_list, dst_list) in edges.items():
                    # src_tensor = torch.tensor(src_list, dtype=torch.int32)
                    # dst_tensor = torch.tensor(dst_list, dtype=torch.int32)
                    # edges[edge_type_str] = (src_tensor, dst_tensor)
                    src_array = np.array(src_list)
                    dst_array = np.array(dst_list)
                    edges[edge_type_str] = (src_array, dst_array)

                # Create heterogeneous graph
                G = dgl.heterograph(edges)

                # print(G)

                # 添加节点属性和边属性
                for ntype in G.ntypes:
                    if ntype == 'keep':
                        nodes = keep_node
                    elif ntype == 'del':
                        nodes = del_node
                    elif ntype == 'add':
                        nodes = add_node
                    # print(
                    #     f"Node type: {ntype}, Number of nodes: {len(nodes)}, Number of attributes: {len(nodes[0])}")  # 添加这行打印语句
                    # G.nodes[ntype].data['inp'] = torch.tensor(nodes)
                    G.nodes[ntype].data['inp'] = torch.stack(nodes, dim=0)

                G.node_dict = {}
                G.edge_dict = {}
                for ntype in G.ntypes:
                    G.node_dict[ntype] = len(G.node_dict)
                for etype in G.canonical_etypes:
                    G.edge_dict[etype] = len(G.edge_dict)
                    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]
                # for etype in G.etypes:
                #     G.edge_dict[etype] = len(G.edge_dict)
                    # G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]

                graph_list.append(G)
                labels.append(graph['label'])
    # dataset = GraphDataset(graph_list, labels)
    # if (0 == len(dataset)):
    #     print(f'[ERROR] Fail to load data from {path}')

    # return dataset, files
    return graph_list, labels, files

# main
def main():
    graph_list, labels, files = GetDataset(path=testPath)
    train_ids = []
    test_ids = []
    # with open('dataset/data_split/new_commit_cross_train.txt', 'r') as file:
    with open('dataset/data_split/new_commit_time_train.txt', 'r') as file:
        line = file.readline()
        while line:
            train_ids.append(line.replace('\n', ''))
            line = file.readline()
    # with open('dataset/data_split/new_commit_cross_test.txt', 'r') as file:
    with open('dataset/data_split/new_commit_time_test.txt', 'r') as file:
        line = file.readline()
        while line:
            test_ids.append(line.replace('\n', ''))
            line = file.readline()

    train_indices = []
    test_indices = []
    for id in files:
        if id in train_ids:
            train_indices.append(files.index(id))
        if id in test_ids:
            test_indices.append(files.index(id))
    # random.seed(42)

    # Create an index list
    # num_graphs = len(graph_list)
    # index_list = list(range(num_graphs))

    # Shuffle the index list
    # random.shuffle(index_list)

    # Define the split ratios (e.g., 80% train, 10% validation, 10% test)
    # train_ratio = 0.8
    # val_ratio = 0.1
    # test_ratio = 0.1

    # Calculate the sizes of each split
    # train_size = int(num_graphs * train_ratio)
    # val_size = int(num_graphs * val_ratio)

    # Split the index list into train, validation, and test sets
    # train_indices = index_list[:train_size]
    # val_indices = index_list[train_size:train_size + val_size]
    # test_indices = index_list[train_size + val_size:]

    # Use the indices to get the corresponding graphs and labels
    train_graphs = [graph_list[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]


    test_graphs = [graph_list[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_files = [files[i] for i in test_indices]

    n_out = 2  # 输出类别数
    num_node_types = 3  # 节点类型数
    # num_edge_types = 27  # 边类型数
    num_edge_types = 18  # 边类型数
    embedding_dim = 52  # 输入维度
    n_hid = 52  # 隐藏层维度
    n_layers = 3  # 图的层数
    n_heads = 4  # 注意力头的数量

    model = HGTClassfication(n_out, num_node_types, num_edge_types, embedding_dim, n_hid, n_layers, n_heads)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    HGTTrain(model, train_graphs, train_labels, optimizer, _epochs_, test_graphs, test_labels, mdlsPath + f'model_HGT_PDG_CCT5_20240628.pth')
    model.load_state_dict(torch.load(mdlsPath + f'model_HGT_PDG_CCT5_20240628.pth')['model_state_dict'])

    testAcc, testPred, testLabel = HGTTest(model, test_graphs, test_labels)

    filename = logsPath + '/test_results_hgt.txt'
    fp = open(filename, 'w')
    fp.write(f'filename,prediction\n')
    for i in range(len(test_files)):
        fp.write(f'{test_files[i]},{testPred[i]}\n')
    fp.close()

    return

if __name__ == '__main__':
    logfile = 'test.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))

    # CCT5
    # extruct_and_constructgraphs()
    main()



