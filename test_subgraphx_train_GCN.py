import sys

import torch
from util.config import args
# from util.models import GCN
# from util.metrics import accuracy, softmax_cross_entropy
from util.graphprocessor_subgraphx import YANCFG_subgraphx

# import networkx as nx
# import tensorflow as tf

# for writing results
# from tensorboardX import SummaryWriter
# from tqdm import tqdm

# from dig.xgraph.dataset import SynGraphDataset
# from dig.xgraph.models import *
# import torch
# from torch_geometric.data import DataLoader
# from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
import os.path as osp
import os

from gcn_model import GCN_3l
from train_gcn_model import TrainModel

# -------------------------------------
# main code to convert and create the dataset
# -------------------------------------

def train_model(path):
    # args.d = 13  # the number of features (fixed)
    # args.c = 13  # the number of classes (fixed)
    # args.batch_size = int(arguments[0])  # batch size
    args.path = str(path)  # the path to load the data
    args.malware_list = {
        'Bagle': 0,
        'Benign': 1,
        'Bifrose': 2,
        'Hupigon': 3,
        'Koobface': 4,
        'Ldpinch': 5,
        'Lmir': 6,
        'Rbot': 7,
        'Sdbot': 8,
        'Swizzor': 9,
        'Vundo': 10,
        'Zbot': 11,
        'Zlob': 12
    }

    # dataset = SynGraphDataset("./datsets", 'BA_2Motifs')
    # print(type(dataset))
    data_loader = YANCFG_subgraphx()
    train, num_samples = data_loader.load_yancfg_data_subgraphx(args.path, 'padded_train', args.malware_list, edge_weights=False)

    print("get test samples")
    test, num_samples = data_loader.load_yancfg_data_subgraphx(args.path, 'padded_test', args.malware_list)    

    # from gcn_model import GCN_3l
    model = GCN_3l(model_level='graph', dim_node=13, dim_hidden=[1024, 512, 128], num_classes=13)
    # model = GCN_2l(model_level='graph', dim_node=dim_node, dim_hidden=100, num_classes=num_classes)

    # the model itself will be different
    from train_gcn_model import TrainModel

    print("\n> created graph classification model", model)
    dataloader_params = {
        'batch_size': 10,
        'random_split_flag': False,
        'data_split_ratio': [0.9, 0.05, 0.05],
        'seed':100
    }
    
    train_params = {
        'num_epochs': 100,
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path = "/home/jherath1/project_3/prj3_CFGExplainer/0_subgraphx/2_comparison_subgraphX"
    trainer = TrainModel(
        model=model,
        dataset_train=train,
        dataset_test=test,
        device=device,
        graph_classification=True,
        save_dir=os.path.join(path, "models"),
        save_name="gcn_test_CFG_sample1_v2",
        dataloader_params=dataloader_params
    )

    trainer.train(
        train_params=train_params,
        optimizer_params=optimizer_params
    )

    _, _, _ = trainer.test()
    # will also need to save the model
    return

path = "../../../y_datasets/YANCFG_sample1"
train_model(path)
