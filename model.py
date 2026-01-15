import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, static_feature_dim=9, hidden_dim=64, num_layers=1, graph_input_dim=64, graph_hidden_dim=64, graph_output_dim=1, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(graph_input_dim, graph_hidden_dim)
        self.conv2 = GCNConv(graph_hidden_dim, graph_output_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.l1 = nn.Linear(hidden_dim+static_feature_dim, graph_input_dim)

    def forward(self, x, sdi, edge_index, batch):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        x = torch.hstack([last_hidden, sdi])
    
        x = self.l1(x)

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        x = x.view(-1)
    
        return x
