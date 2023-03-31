
import torch
from torch_geometric.nn.conv import GENConv
from torch_geometric.nn import Linear, global_add_pool, DeepGCNLayer
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, SELU, LayerNorm, ReLU
from torch_geometric.utils import add_self_loops, scatter
from utils.pooling import GlobalAddPooling
from utils.sigmoid import Sigmoid

class GCN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_channels, pool_dim, fully_connected_channels, output_channels):
        super(GCN, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.lin = torch.nn.ModuleList()
        self.node_encoder = torch.nn.ModuleList()
        self.edge_encoder = torch.nn.ModuleList()

        # hidden_channels = [node_feature_dim] + hidden_channels

        for i in range(len(hidden_channels)):
            conv = GENConv(hidden_channels[i], pool_dim, edge_dim = hidden_channels[i], msg_norm=True, learn_msg_scale=True)
            norm = LayerNorm(pool_dim)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block="plain", dropout=.5) 
            self.layers.append(layer)
            
            self.lin.append(Linear(hidden_channels[i], pool_dim))

            self.node_encoder.append(Linear(node_feature_dim, hidden_channels[i]))
            self.edge_encoder.append(Linear(edge_feature_dim, hidden_channels[i]))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(pool_dim, fully_connected_channels[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[0]),
            torch.nn.Dropout(p=.47),

            torch.nn.Linear(fully_connected_channels[0], fully_connected_channels[1]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[1]),
            torch.nn.Dropout(p=.47),

            torch.nn.Linear(fully_connected_channels[1], fully_connected_channels[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[2]),
            torch.nn.Dropout(p=.47),

            torch.nn.Linear(fully_connected_channels[2], output_channels),
            Sigmoid()
        )

        self.activate = SELU()
        self.pool = GlobalAddPooling()

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings 
        readout = 0

        for layer, lin, node_encoder, edge_encoder in zip(self.layers, self.lin, self.node_encoder, self.edge_encoder):
            data = node_encoder(x)
            data_edge_attr = edge_encoder(edge_attr)
            # print(edge_attr.shape)
            layer_read = layer(data, edge_index, data_edge_attr)
            readout += F.softmax(layer_read, dim=-1)

        # 2. Readout layer
        x = self.pool(readout, batch)  # [batch_size, hidden_channels]

        # 3. Fully Connected Neural Net
        x = self.classifier(x)
    
        return x


class GCN2(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_channels, pool_dim, fully_connected_channels, output_channels):
        super(GCN2, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.lin = torch.nn.ModuleList()
        self.node_encoder = torch.nn.ModuleList()
        self.edge_encoder = torch.nn.ModuleList()

        # hidden_channels = [node_feature_dim] + hidden_channels

        for i in range(len(hidden_channels)):
            conv = GENConv(hidden_channels[i], pool_dim, edge_dim = hidden_channels[i], msg_norm=True, learn_msg_scale=True)
            norm = LayerNorm(pool_dim)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block="plain", dropout=.5) 
            self.layers.append(layer)
            
            self.lin.append(Linear(hidden_channels[i], pool_dim))

            self.node_encoder.append(Linear(node_feature_dim, hidden_channels[i]))
            self.edge_encoder.append(Linear(edge_feature_dim, hidden_channels[i]))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(pool_dim, fully_connected_channels[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[0]),
            torch.nn.Dropout(p=.47),

            torch.nn.Linear(fully_connected_channels[0], fully_connected_channels[1]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[1]),
            torch.nn.Dropout(p=.47),

            torch.nn.Linear(fully_connected_channels[1], output_channels),
            Sigmoid()
        )

        self.activate = SELU()
        self.pool = GlobalAddPooling()

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings 
        readout = 0

        for layer, lin, node_encoder, edge_encoder in zip(self.layers, self.lin, self.node_encoder, self.edge_encoder):
            data = node_encoder(x)
            data_edge_attr = edge_encoder(edge_attr)
            # print(edge_attr.shape)
            layer_read = layer(data, edge_index, data_edge_attr)
            readout += F.softmax(layer_read, dim=-1)

        # 2. Readout layer
        x = self.pool(readout, batch)  # [batch_size, hidden_channels]

        # 3. Fully Connected Neural Net
        x = self.classifier(x)
    
        return x

class GCN2Regressor(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_channels, pool_dim, fully_connected_channels, output_channels):
        super(GCN2Regressor, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.lin = torch.nn.ModuleList()
        self.node_encoder = torch.nn.ModuleList()
        self.edge_encoder = torch.nn.ModuleList()

        # hidden_channels = [node_feature_dim] + hidden_channels

        for i in range(len(hidden_channels)):
            conv = GENConv(hidden_channels[i], pool_dim, edge_dim = hidden_channels[i], msg_norm=True, learn_msg_scale=True)
            norm = LayerNorm(pool_dim)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block="plain", dropout=.5) 
            self.layers.append(layer)
            
            self.lin.append(Linear(hidden_channels[i], pool_dim))

            self.node_encoder.append(Linear(node_feature_dim, hidden_channels[i]))
            self.edge_encoder.append(Linear(edge_feature_dim, hidden_channels[i]))

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(pool_dim, fully_connected_channels[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[0]),
            torch.nn.Dropout(p=.47),

            torch.nn.Linear(fully_connected_channels[0], fully_connected_channels[1]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[1]),
            torch.nn.Dropout(p=.47),

            torch.nn.Linear(fully_connected_channels[1], output_channels)
        )

        self.activate = SELU()
        self.pool = GlobalAddPooling()

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings 
        readout = 0

        for layer, lin, node_encoder, edge_encoder in zip(self.layers, self.lin, self.node_encoder, self.edge_encoder):
            data = node_encoder(x)
            data_edge_attr = edge_encoder(edge_attr)
            # print(edge_attr.shape)
            layer_read = layer(data, edge_index, data_edge_attr)
            readout += F.softmax(layer_read, dim=-1)

        # 2. Readout layer
        x = self.pool(readout, batch)  # [batch_size, hidden_channels]

        # 3. Fully Connected Neural Net
        x = self.regressor(x)
    
        return x
