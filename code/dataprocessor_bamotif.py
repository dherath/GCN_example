import pickle
import torch
from torch_geometric.data import Data


class example_BAMOTIF:

    """
    dataprocessor class for example BA2MOTIF dataset
    information of dataset can be found in PGExplainer paper
    : https://openreview.net/pdf?id=tt04glo-VrT
    
    > there are 1000 graphs with each graph having 25 nodes
    > there are no edge features, the graphs have two classes 0,1
    > this is a graph classification task

    > pytorch geometric requires graphs parsed as an edge_list
    > the function parse_single_graph() is where it is done
    """

    def __init__(self):
        return

    def load_pickle(self, filename):
        """
        loads a pickle
        ----------------
        Args
        filename (str): path to file
        ----------------
        Returns
        data : the pickled data in list format
        """
        data = None
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
        return data

    def parse_single_graph(self, adj, feat, label):
        """
        parses a single graph into pytorch-geometric format
        the libarary reuqires graphs parsed as an edge_list
        ----------------
        Args
        adj (list): 25x25 adjacency matrix
        feat (list): [25, #feats] feature matrix
        label (int): 0/1 for class label
        ----------------
        Returns
        x (torch.tensor): the features
        edge_list (torch.tensor): the edges list [[i,...], [j,...]] for i->j edge
        y (torch.tensor): the labels
        """
        # print(feat)
        x = torch.tensor(feat, dtype=torch.float32)
        # print(label)
        label_id = 0 if label[0] == 1 else 1
        y = torch.tensor(label_id, dtype=torch.long)  # label always has to be long, for higher precision
        edge_list, u, v = [], [], []
        
        # loop throught he adj. matrix and fill in edges as
        # for an i->j edge, [[i,...], [j,...]]

        # since graphs are small for this dataset (25 nodes)
        # I will loop through the entire matrix manually
        # for larger graphs better to convert to coo_matrix first
        
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i][j] == 1:
                    u.append(i)
                    v.append(j)

        edge_list.append(u)
        edge_list.append(v)
        # print(x.shape)
        # print(y.shape)
        # print(edge_list)
        # sys.exit()
        edge_list = torch.tensor(edge_list, dtype=torch.float32)
        return x, edge_list, y

    def parse_all_data(self, filename):
        """
        parses all data into required format
        ----------------
        Args
        filename (str): the filename to load data
        ----------------
        Returns
        dataset (list): list of pytorch-geometric Data samples
        """
        data = self.load_pickle(filename)
        # the data is already stored as [adjacency-matrices, feats, labels]
        adj_matrices = data[0]
        feats = data[1]
        labels = data[2]

        dataset = []  # will store the datset here

        # loop through all samples and parse
        for adj, feat, label in zip(adj_matrices, feats, labels):
            # x = node features, y = label
            # if there are edge_weights, then parse_single_graph() must be modified
            x, edge_list, y = self.parse_single_graph(adj, feat, label)
            
            # if there are edge weights
            # then Data(x=x, edge_index=edge_list, edge-weight=wdge_weight, y=y)
            dataset.append(Data(x=x, edge_index=edge_list, y=y))

        return dataset

    
