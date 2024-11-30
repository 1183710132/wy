import torch
from torch_geometric.nn import GCNConv

class WorkflowGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WorkflowGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = torch.relu(x)  # 激活函数
        # 第二层图卷积
        x = self.conv2(x, edge_index)
        return x
