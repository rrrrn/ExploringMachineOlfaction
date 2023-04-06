import torch
from torch_geometric.nn import global_add_pool


class GlobalAddPooling(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, readout, batch):
        return global_add_pool(readout, batch=batch)
