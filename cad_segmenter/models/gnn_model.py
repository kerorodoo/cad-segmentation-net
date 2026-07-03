import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


class CADFeatureSegmenter(torch.nn.Module):
    """Deep Geometric Learning Module for face-level semantic segmentation using GATv2."""

    def __init__(self, in_channels: int, num_classes: int, heads: int = 4):
        """Initializes the GATv2 layers with multi-head attention and edge-attribute awareness."""
        super().__init__()
        # First layer projects to 32 hidden units per head (output: 32 * 4 = 128 hidden channels)
        self.conv1 = GATv2Conv(in_channels, 32, heads=heads, edge_dim=1)

        # Second layer projects back to 128 channels using a single final head
        self.conv2 = GATv2Conv(32 * heads, 128, heads=1, concat=False, edge_dim=1)

        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass utilizing both node features and edge dihedral angles.

        Args:
            data: PyTorch Geometric Data object containing:
                  - x: Node features [num_faces, 6]
                  - edge_index: Topological face adjacency index [2, num_edges]
                  - edge_attr: Dihedral angle of boundaries [num_edges, 1]

        Returns:
            Log-probabilities of each face belonging to each class.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Convolutions with attention weighted by node features and dihedral transition angles
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.05, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        return F.log_softmax(self.classifier(x), dim=1)
