import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from cad_segmenter.controllers.train_controller import TrainController
from cad_segmenter.models.gnn_model import CADFeatureSegmenter


def get_node_embeddings(model: CADFeatureSegmenter, data) -> np.ndarray:
    """Extracts node embeddings just before the classification projection layer."""
    backbone = model.net
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

    if model.backbone_name == "gcn":
        h = backbone.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.05, training=False)
        h = backbone.conv2(h, edge_index)
        return F.relu(h).cpu().numpy()

    elif model.backbone_name == "gatv2":
        h = backbone.conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=0.05, training=False)
        h = backbone.conv2(h, edge_index, edge_attr)
        return F.relu(h).cpu().numpy()

    elif model.backbone_name == "gine":
        h = backbone.conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=0.05, training=False)
        h = backbone.conv2(h, edge_index, edge_attr)
        return F.relu(h).cpu().numpy()

    elif model.backbone_name == "graphgps":
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        h = backbone.project_in(x)
        h = backbone.gps_conv1(h, edge_index, batch=batch, edge_attr=edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=0.05, training=False)
        h = backbone.gps_conv2(h, edge_index, batch=batch, edge_attr=edge_attr)
        return F.relu(h).cpu().numpy()

    else:
        raise ValueError(f"Unknown backbone name: {model.backbone_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNN Overfitting and Z-Coordinate Leakage Diagnostic Script"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["gcn", "gatv2", "gine", "graphgps"],
        default="gatv2",
        help="GNN backbone to evaluate (default: gatv2).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="cad_segmenter/models/weights/pretrained_segmenter.pth",
        help="Path to GNN weights (default: cad_segmenter/models/weights/pretrained_segmenter.pth).",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/val",
        help="Path to validation directory (default: data/val).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/overfitting_visual_proof.png",
        help="Path to output plot (default: plots/overfitting_visual_proof.png).",
    )
    args = parser.parse_args()

    # 1. Verify weights file exists
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found at: {args.weights}")

    print(f"Loading validation graphs from: {args.val_dir}")
    controller = TrainController()
    val_graphs = controller._load_graphs_from_directory(args.val_dir)
    print(f"Successfully loaded {len(val_graphs)} validation graphs.")

    # 2. Instantiate and load model
    print(f"Instantiating {args.backbone.upper()} model and loading weights...")
    model = CADFeatureSegmenter(in_channels=6, num_classes=6, backbone=args.backbone)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    all_embeddings = []
    all_labels = []
    all_z_coords = []

    # 3. Extract node embeddings and Z-heights
    print("Extracting intermediate node embeddings and physical Z coordinates...")
    max_graphs = min(50, len(val_graphs))
    with torch.no_grad():
        for i in range(max_graphs):
            data = val_graphs[i]
            try:
                embeddings = get_node_embeddings(model, data)
                all_embeddings.append(embeddings)
                all_labels.append(data.y.cpu().numpy())
                # Z-Height of centroid is index 5 in feature vector
                all_z_coords.append(data.x[:, 5].cpu().numpy())
            except Exception as e:
                print(f"Warning: skipped graph {i} due to extraction error: {e}")

    if not all_embeddings:
        print("Error: No node embeddings could be extracted. Exiting.")
        return

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    z_coords = np.concatenate(all_z_coords, axis=0)

    # 4. Project high-dimensional embeddings to 2D using t-SNE
    print(
        f"Reducing {embeddings.shape[0]} node embeddings (dim={embeddings.shape[1]}) to 2D via t-SNE..."
    )
    # Adjust perplexity for small embedding sizes safely
    perp = min(30, max(2, len(embeddings) // 10))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 5. Build side-by-side diagnostic figures
    print("Generating dual-panel visual proof figure...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: Colored by Class Label
    scatter1 = ax1.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.8,
        edgecolors="none",
    )
    ax1.set_title("GNN Embedding Space: Colored by Ground Truth Class")
    ax1.set_xlabel("t-SNE Axis 1")
    ax1.set_ylabel("t-SNE Axis 2")
    ax1.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(scatter1, ax=ax1, label="Class ID (0-5)")

    # Right plot: Colored by Absolute Z coordinate (Height)
    scatter2 = ax2.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=z_coords,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )
    ax2.set_title("GNN Embedding Space: Colored by Absolute Z-Coordinate (Height)")
    ax2.set_xlabel("t-SNE Axis 1")
    ax2.set_ylabel("t-SNE Axis 2")
    ax2.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(scatter2, ax=ax2, label="Centroid Z (mm)")

    plt.suptitle(
        f"Diagnostic Report: Overfitting and Feature Leakage Visual Proof ({args.backbone.upper()} Model)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"SUCCESS: Diagnostic visual proof figure saved to: {args.output}")


if __name__ == "__main__":
    main()
