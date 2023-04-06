import torch
from torch_geometric.nn.conv import GCNConv
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, SELU
from torch_geometric.utils import add_self_loops, scatter
from utils.pooling import GlobalAddPooling
from utils.sigmoid import Sigmoid


class GNN(torch.nn.Module):
    """
    Graph Neural Network for classification
    four semi-supervised graph convolution layers (GCN) from <https://arxiv.org/pdf/1609.02907.pdf>
    max graph pool layer followed by a three-layer fully-connected prediction networks
    """

    def __init__(
        self,
        input_channels,
        hidden_channels,
        pool_dim,
        fully_connected_channels,
        output_channels,
    ):
        """
        Args:
            input_channels (int): Size of each input sample
            output_channels (int): Size of each output sample
            hidden_channels (tuple or list): input and output size of each GCN layer
            pool_dim (int): size of the pooling layer
            fully_connected_channels (tuple or list): dimension for fully-connected prediction layers
        """
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = GCNConv(hidden_channels[1], hidden_channels[2])
        self.conv4 = GCNConv(hidden_channels[2], hidden_channels[3])

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(pool_dim, fully_connected_channels[0]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[0]),
            torch.nn.Dropout(p=0.47),
            torch.nn.Linear(fully_connected_channels[0], fully_connected_channels[1]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[1]),
            torch.nn.Dropout(p=0.47),
            torch.nn.Linear(fully_connected_channels[1], fully_connected_channels[2]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fully_connected_channels[2]),
            torch.nn.Dropout(p=0.47),
            torch.nn.Linear(fully_connected_channels[2], output_channels),
            Sigmoid(),
            torch.nn.Softmax(dim=1),
        )

        self.lin_1 = torch.nn.Linear(hidden_channels[0], pool_dim)
        self.lin_2 = torch.nn.Linear(hidden_channels[1], pool_dim)
        self.lin_3 = torch.nn.Linear(hidden_channels[2], pool_dim)
        self.lin_4 = torch.nn.Linear(hidden_channels[3], pool_dim)
        self.activate = SELU()
        self.pool = GlobalAddPooling()

    def forward(self, atom_feat, edge_index, batch):
        """
        update nodes using atom feature and edge index information for every batch
        """
        # 1. Obtain node embeddings
        readout = 0

        x = self.conv1(atom_feat, edge_index)
        x = self.activate(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.lin_1(x), dim=-1)

        x = self.conv2(x, edge_index)
        x = self.activate(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.lin_2(x), dim=-1)

        x = self.conv3(x, edge_index)
        x = self.activate(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.lin_3(x), dim=-1)

        x = self.conv4(x, edge_index)
        x = self.activate(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.lin_4(x), dim=-1)

        # 2. Readout layer
        x = self.pool(readout, batch)  # [batch_size, hidden_channels]

        # 3. Fully Connected Neural Net
        x = self.classifier(x)

        return x

    @staticmethod
    def max_graph_pool(x, edge_index):
        edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index
        x = scatter(x[row], col, dim=0, reduce="max")
        return x
