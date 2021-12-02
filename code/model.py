import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from torch_geometric.data.batch import Batch


class GNNBasic(torch.nn.Module):

    """
    Parent class for GNNs
    """
    
    def __init__(self):
        super().__init__()

    def arguments_read(self, *args, **kwargs):
        """
        loads the graph samples for forward pass in models
        """
        data: Batch = kwargs.get('data') or None
        if not data:
            # this condition is for testing phase,
            # if a single sample is given for the forward pass
            x, edge_index, batch = args[0].x, args[0].edge_index, args[0].batch
        else:
            # this is when a mini-batch of data is loaded for forward pass
            x, edge_index, batch = data.x, data.edge_index, data.batch
        # note: edge_index == edge_list, the pytorch-geometric library calls it the edge_index
        # note: this function will need to change if edge attributes are available
        return x, edge_index, batch


class GCN(GNNBasic):

    """
    A GCN based GNN model with dropout and readout functions
    The model is for graph classification tasks
    """

    def __init__(self, dim_node, dim_hidden, num_classes, dropout_level):
        """
        dim_node (int): the #features per node
        dim_hidden (list): the list of hidden sizes after each convolution step
        num_classes (int): the number of classes
        dropout_level (float): the dropout probability
        """
        super().__init__()
        num_layer = len(dim_hidden)

        self.conv1 = gnn.GCNConv(dim_node, dim_hidden[0])

        # append the conv layers
        # if other types of conv layers are needed this is where to changes them
        # if edge_weights are availabe, must change it here (must refer pytorch-geometric documentation)
        layers = []
        for i in range(num_layer - 1):
            layers.append(gnn.GCNConv(dim_hidden[i], dim_hidden[i + 1]))
        self.convs = nn.ModuleList(layers)
        
        # can change the activation functions as required
        # in case the data contains negative numbers then Relu()
        # might not be ~good
        
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )

        self.readout = GlobalMeanPool()

        # can also use an additional Linear layer before predictions as follows:
        # self.ffn = nn.Sequential(*(
        #         [nn.Linear(dim_hidden[-1], dim_hidden[-1])] +
        #         [nn.ReLU(), nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
        # ))

        self.ffn = nn.Sequential(*(
            [nn.Dropout(p=dropout_level), nn.Linear(dim_hidden[-1], num_classes)]
        ))
        return

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Does a single forward pass for the complete model
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        # 1. first conv. pass -> uses the original data sample
        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            # 2. iteratively do GCN convolution
            post_conv = relu(conv(post_conv, edge_index))

        # 3. use the readout (i.e., pooling)
        out_readout = self.readout(post_conv, batch)

        # 4. the class probabilities
        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Auxilary function if node embeddings are required seperately
        works similar to the forward pass above
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))
        return post_conv


# ----------------------------
# The following are Pooling layers for the readout functions
# Currently I used to methods
# 1. MeanPool: Takes the mean for node embeddings per node
# 2. Identity: Does not change the node emebddgins
# ----------------------------


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)


class IdenticalPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x
