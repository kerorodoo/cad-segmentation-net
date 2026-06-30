import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class CADFeatureSegmenter(torch.nn.Module):
    """Deep Geometric Learning Module for face-level semantic segmentation."""

    def __init__(self, in_channels: int, num_classes: int):
        """Initializes the GNN layers and classification head."""
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 128)
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            data: PyTorch Geometric Data object containing node features (x)
                  and topological links (edge_index).

        Returns:
            Log-probabilities of each face belonging to each class.
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return F.log_softmax(self.classifier(x), dim=1)
