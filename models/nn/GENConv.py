import torch
from torch_geometric.nn.conv import GENConv
from torch_geometric.nn import Linear, global_add_pool, DeepGCNLayer
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, SELU, LayerNorm, ReLU
from torch_geometric.utils import add_self_loops, scatter
from utils.pooling import GlobalAddPooling
from utils.sigmoid import Sigmoid


class GCN2Regressor(torch.nn.Module):
    """
    Graph Neural Network for regression
    a customizable number of The GENeralized Graph Convolution (GENConv) layer from <https://arxiv.org/pdf/2006.07739.pdf>
    max graph pool layer followed by a two-layer fully-connected prediction networks
    """

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        hidden_channels,
        pool_dim,
        fully_connected_channels,
        output_channels,
    ):
        """
        Args:
            node_feature_dim (int): Size of node featrues
            edge_feature_dim (int): Size of edge features
            hidden_channels (tuple or list): output size of each GEN layer
            pool_dim (int): size of the pooling layer
            fully_connected_channels (tuple or list): dimension for fully-connected prediction layers
        """
        super(GCN2Regressor, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.lin = torch.nn.ModuleList()
        self.node_encoder = torch.nn.ModuleList()
        self.edge_encoder = torch.nn.ModuleList()

        for i in range(len(hidden_channels)):
            conv = GENConv(
                hidden_channels[i],
                pool_dim,
                edge_dim=hidden_channels[i],
                msg_norm=True,
                learn_msg_scale=True,
            )
            norm = LayerNorm(pool_dim)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block="plain", dropout=0.47)
            self.layers.append(layer)

            self.lin.append(Linear(hidden_channels[i], pool_dim))

            self.node_encoder.append(Linear(node_feature_dim, hidden_channels[i]))
            self.edge_encoder.append(Linear(edge_feature_dim, hidden_channels[i]))

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(pool_dim, fully_connected_channels[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[0]),
            torch.nn.Dropout(p=0.47),
            torch.nn.Linear(fully_connected_channels[0], fully_connected_channels[1]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[1]),
            torch.nn.Dropout(p=0.47),
            torch.nn.Linear(fully_connected_channels[1], output_channels),
        )

        self.pool = GlobalAddPooling()

    def forward(self, atom_feat, edge_index, edge_attr, batch):
        """
        update nodes using atom feature, edge feature, and edge index information for every batch
        """
        # 1. Obtain node embeddings
        readout = 0

        for layer, lin, node_encoder, edge_encoder in zip(
            self.layers, self.lin, self.node_encoder, self.edge_encoder
        ):
            data = node_encoder(atom_feat)
            data_edge_attr = edge_encoder(edge_attr)
            layer_read = layer(data, edge_index, data_edge_attr)
            readout += F.softmax(layer_read, dim=-1)

        # 2. Readout layer
        x = self.pool(readout, batch)  # [batch_size, hidden_channels]

        # 3. Fully Connected Neural Net
        x = self.regressor(x)

        return x
