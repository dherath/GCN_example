# from dig.xgraph.dataset import SynGraphDataset
# from dig.xgraph.models import *
import sys
import torch
import pickle
# from torch_geometric.data import DataLoader
# from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
# import os.path as osp
import os
from scipy.sparse import coo_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_pickle(filename):
    """
    loads pickle data
    """
    data = None
    with open(filename, 'rb') as fp:
        print(filename)
        data = pickle.load(fp)
    return data


def process_data(filename):
    """
    processes the dataset, pytorch gemoteric needs data in coo-format
    the tensor dataloader will need = node_feat, edge_indices, edge_weights (optional), class_labels
    """

    # the example dataset has [adjacency-matrix, featues, labels] stored in seperate lists
    data = load_pickle(filename)
    adjs = data[0]  # list of adjacency matricex
    feats = data[1]
    labels = data[2]

    # will loop through all data and convert it to tensor format
    for adj, feat, label in zip(adjs, feats, labels):
        row, col, edge_attr = adj.t().coo()
        return
    
    
    
    return


def main():
    """
    trains a GCN model
    """
    
    filename = "./datasets/BA_2Motifs.pkl"
    data = load_pickle(filename)

    # preprocessing the data
    # print(data[0].shape)
    # labels =
    adj = data[0][0]
    coo = coo_matrix(adj)
    # edge_index = torch.utils.from_scipy_sparse_matrix(coo)
    # print(edge_index)
    print(coo.shape)
    return

main()
sys.exit()


# 1. getting the dataset

# print("Node classification example!!! [BA_shapes]")
print("> SubgraphX graph classification")

dataset = SynGraphDataset('./datasets', 'BA_2Motifs')
dataset.data.x = dataset.data.x.to(torch.float32)
dataset.data.x = dataset.data.x[:, :1]
dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
num_classes = dataset.num_classes

print("> loaded data Synthetic BA_2Motifs dim_node = {}, dim_edge = {}, num_classes = {}".format(dim_node, dim_edge, num_classes))

# print(dataset.data)

# print(len(dataset))
# print(len(dataset.data))

# format of the data (will need to process the data like this)
print(dataset[0].x.shape)  # the node features
print(dataset[0].edge_index.shape)  # the edge list
print(dataset[0].y)  # the graph class label

# 2. training a graph classifier

# 2.1. creating the graph model
from gcn_model import GCN_3l
model = GCN_3l(model_level='graph', dim_node=dim_node, dim_hidden=[1024, 512, 128], num_classes=num_classes)
# model = GCN_2l(model_level='graph', dim_node=dim_node, dim_hidden=100, num_classes=num_classes)

# the model itself will be different

print("\n> created graph classification model", model)

from train_gcn_model import TrainModel

# 2.2.1. must do this step?
dataset.data.x = dataset.data.x.float()
dataset.data.y = dataset.data.y.squeeze().long()

# -> set dataloader params
dataloader_params = {
    'batch_size': 128,
    'random_split_flag': True,
    'data_split_ratio': [0.8, 0.1, 0.1],
    'seed':100
}

train_params = {
    'num_epochs': 10,
    'num_early_stop': 100,
    'milestones': None,
    'gamma': None
}

optimizer_params = {
    'lr': 0.0001,
    'weight_decay': 5e-4
}

# weight decay is a regularizing term, L2 norm
# read about this

path = "/home/jherath1/project_3/prj3_CFGExplainer/0_subgraphx/1_initial_test"
trainer = TrainModel(
    model=model,
    dataset=dataset,
    device=device,
    graph_classification=True,
    save_dir=os.path.join(path, "models"),
    save_name="gcn_test_BA2Motif_v2",
    dataloader_params=dataloader_params
)

trainer.train(
    train_params=train_params,
    optimizer_params=optimizer_params
)

_, _, _ = trainer.test()
# will also need to save the model


# 3. running the explainer
from dig.xgraph.method import SubgraphX

print("\n> calling subgraphX")
subgraphx = SubgraphX(model=model, num_classes=num_classes, device=device,
                      explain_graph=True)
x = dataset[0].x.to(device)
edge_index = dataset[0].edge_index.to(device)
import time

t1 = time.time()
_, explanation_results, related_preds = subgraphx(x, edge_index, max_nodes=10)
t2 = time.time()

print("time diff (s)", str(t2 - t1))

# print("\n exp.results:")
# print(explanation_results)

# print("\n rrelated_preds:")
# print(related_preds)

# 4. visualize the results

print("inputs:")
print(x.shape, edge_index.shape)

# print("predictions")
# print(explanation_results.shape)
# print(len(related_preds))

y = dataset[0].y  # the label for the prediction

print(y)

#print(explanation_results[y.item()])

exp_result = explanation_results[y.item()]
result = subgraphx.read_from_MCTSInfo_list(exp_result)

# see if we get a result for each node?
# not the results are for each MCTS iterations
print("#results:", len(result))


for idx, val in enumerate(result):
    print(idx, val.data, len(val.coalition), val.W, val.N, val.P)
    if idx > 50:
        break

print()

def find_first_match(results, max_nodes):
    # ToDo: update code with a tolerance, to identify
    # the best graph with the most number of nodes
    # then will have to use that specific graph again iteratively?

    # 
    """ return the highest reward tree_node with its subgraph is ~= max_nodes """
    results = sorted(results, key=lambda x: len(x.coalition))

    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        if abs(len(x.coalition) - max_nodes) <= 2 and x.P > result_node.P:
            result_node = x
    return result_node

result_node = find_first_match(result, 10)
print(result_node)

print(len(result_node.coalition))


print()
print("coalition:", result_node.coalition)
print("score for selection:", result_node.P)
# print(result.)
