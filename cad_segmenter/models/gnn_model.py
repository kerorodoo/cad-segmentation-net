import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINEConv, GPSConv
from torch_geometric.data import Data


class GCNBackbone(torch.nn.Module):
    """Isotropic Graph Convolutional Network (GCN) backbone."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 128)
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.classifier(x)


class GATv2Backbone(torch.nn.Module):
    """Anisotropic Graph Attention Network v2 (GATv2) backbone with edge-attribute awareness."""

    def __init__(self, in_channels: int, num_classes: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, 32, heads=heads, edge_dim=1)
        self.conv2 = GATv2Conv(32 * heads, 128, heads=1, concat=False, edge_dim=1)
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        return self.classifier(x)


class GINEBackbone(torch.nn.Module):
    """Graph Isomorphism Network (GINE) backbone with MLPs and edge-attribute awareness."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        )
        self.conv1 = GINEConv(self.mlp1, edge_dim=1)

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )
        self.conv2 = GINEConv(self.mlp2, edge_dim=1)

        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        return self.classifier(x)


class GraphGPSBackbone(torch.nn.Module):
    """General, Powerful, and Scalable (GPS) Graph Transformer backbone (Rampášek et al., NeurIPS 2022).

    Combines local GINE message passing with global multi-head self-attention.
    """

    def __init__(
        self, in_channels: int, num_classes: int, channels: int = 64, heads: int = 4
    ):
        super().__init__()
        self.project_in = torch.nn.Linear(in_channels, channels)

        # 1. Local GINE Layer 1 (edge-aware)
        self.local_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, channels),
        )
        local_conv1 = GINEConv(self.local_mlp1, edge_dim=1)

        # 2. GPS Layer 1 (Local GINE + Global Multi-head Self-Attention)
        self.gps_conv1 = GPSConv(
            channels=channels,
            conv=local_conv1,
            heads=heads,
            dropout=0.05,
            attn_type="multihead",
        )

        # 3. Local GINE Layer 2
        self.local_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, channels),
        )
        local_conv2 = GINEConv(self.local_mlp2, edge_dim=1)

        # 4. GPS Layer 2
        self.gps_conv2 = GPSConv(
            channels=channels,
            conv=local_conv2,
            heads=heads,
            dropout=0.05,
            attn_type="multihead",
        )

        # Output project to 128 classification channels
        self.project_out = torch.nn.Linear(channels, 128)
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # GPSConv requires a batch vector for dynamic pooling
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        # Project features to GPS channel space
        x = self.project_in(x)

        # Layer 1
        x = self.gps_conv1(x, edge_index, batch=batch, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.05, training=self.training)

        # Layer 2
        x = self.gps_conv2(x, edge_index, batch=batch, edge_attr=edge_attr)
        x = F.relu(x)

        # Output linear map
        x = self.project_out(x)
        x = F.relu(x)
        return self.classifier(x)


class CADFeatureSegmenter(torch.nn.Module):
    """Unified GNN Wrapper routing messages dynamically to selected backbone."""

    def __init__(self, in_channels: int, num_classes: int, backbone: str = "gatv2"):
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == "gcn":
            self.net = GCNBackbone(in_channels, num_classes)
        elif self.backbone_name == "gatv2":
            self.net = GATv2Backbone(in_channels, num_classes)
        elif self.backbone_name == "gine":
            self.net = GINEBackbone(in_channels, num_classes)
        elif self.backbone_name == "graphgps":
            self.net = GraphGPSBackbone(in_channels, num_classes)
        else:
            raise ValueError(f"Unknown GNN backbone model: {backbone}")

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the selected neural backbone."""
        return F.log_softmax(self.net(data), dim=1)
