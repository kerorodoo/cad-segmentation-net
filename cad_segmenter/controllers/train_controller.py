import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
from multiprocessing import Pool

from cad_segmenter.models.cad_graph import CADGraphModel
from cad_segmenter.models.gnn_model import CADFeatureSegmenter
from cad_segmenter.models.data_factory import (
    generate_single_variant_process,
)
from cad_segmenter.views.console_view import ConsoleView
from cad_segmenter.utils.metrics import SegmentationMetrics
from torch_geometric.data import Data


class TrainController:
    """Coordinates synthetic dataset splitting, parallel GNN training, and evaluation matrix."""

    def __init__(
        self,
        train_dir: str = "data/train",
        val_dir: str = "data/val",
        test_dir: str = "data/synthetic",
    ) -> None:
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

    def bootstrap_and_train(
        self,
        num_variants: int = 500,
        epochs: int = 25,
        use_existing_dataset: bool = False,
    ) -> str:
        """Procedurally bootstraps Train/Val/Test split datasets or loads existing, fits GNN, and runs evaluation."""
        if use_existing_dataset:
            ConsoleView.log_header("Loading Pre-existing Dataset")
            ConsoleView.log_info(f"Loading Training set from: {self.train_dir}")
            train_graphs = self._load_graphs_from_directory(self.train_dir)
            ConsoleView.log_info(f"Loaded {len(train_graphs)} training examples.")

            ConsoleView.log_info(f"Loading Validation set from: {self.val_dir}")
            val_graphs = self._load_graphs_from_directory(self.val_dir)
            ConsoleView.log_info(f"Loaded {len(val_graphs)} validation examples.")

            if not train_graphs:
                raise ValueError(
                    f"No valid pre-existing training data found in {self.train_dir}"
                )
            if not val_graphs:
                raise ValueError(
                    f"No valid pre-existing validation data found in {self.val_dir}"
                )
        else:
            ConsoleView.log_header("Procedural Dataset Division (700 Parts Total)")
            num_train = num_variants
            num_val_test = max(
                3, num_train // 5
            )  # E.g. If train=500, val=100, test=100. Total = 700

            ConsoleView.log_info(
                f"Generating Parallel Training set ({num_train} parts) -> {self.train_dir}"
            )
            train_graphs = self._generate_split_dataset(self.train_dir, num_train)

            ConsoleView.log_info(
                f"Generating Parallel Validation set ({num_val_test} parts) -> {self.val_dir}"
            )
            val_graphs = self._generate_split_dataset(self.val_dir, num_val_test)

            ConsoleView.log_info(
                f"Generating Parallel Test set ({num_val_test} parts) -> {self.test_dir}"
            )
            _ = self._generate_split_dataset(self.test_dir, num_val_test)

        ConsoleView.log_header("GNN Training Loop (6 Classes)")
        model = CADFeatureSegmenter(in_channels=6, num_classes=6)
        self._run_training_loop(model, train_graphs, epochs)

        ConsoleView.log_header("Validation Dataset Matrix Evaluation")
        self._evaluate_validation(model, val_graphs)

        # Ensure weights folder exists
        weights_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "weights"
        )
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, "pretrained_segmenter.pth")

        torch.save(model.state_dict(), weights_path)
        ConsoleView.log_success(f"Pre-trained weights serialized to: {weights_path}")
        return weights_path

    def _generate_split_dataset(self, directory: str, count: int) -> List[Data]:
        """Generates variants in parallel using CPU Pool and parses them to PyG Data."""
        pool_args = [(directory, i) for i in range(1, count + 1)]

        # Procedurally model geometries in parallel across available CPU cores
        with Pool() as p:
            results = p.map(generate_single_variant_process, pool_args)

        graphs: List[Data] = []
        for step_path, labels_path in results:
            try:
                graph = self._prepare_graph_data(step_path, labels_path)
                graphs.append(graph)
            except Exception as e:
                ConsoleView.log_warning(f"Skipped parsing graph due to: {e}")

        return graphs

    def _prepare_graph_data(self, step_path: str, labels_path: str) -> Data:
        """Parses a physical STEP file and attaches ground-truth face labels."""
        model = CADGraphModel(step_path)
        graph = model.extract_graph_tensors()

        with open(labels_path, "r") as f:
            labels_map = json.load(f)

        num_nodes = graph.x.size(0)
        y = torch.zeros(num_nodes, dtype=torch.long)

        # Align JSON index labels with face node indices
        for f_idx_str, label in labels_map.items():
            idx = int(f_idx_str)
            if idx < num_nodes:
                y[idx] = label

        graph.y = y
        return graph

    def _load_graphs_from_directory(self, directory: str) -> List[Data]:
        """Loads already prepared datasets (STEP and JSON labels) from a directory into PyG Data objects."""
        import glob

        labels_pattern = os.path.join(directory, "*_labels.json")
        labels_files = glob.glob(labels_pattern)

        graphs: List[Data] = []
        for labels_path in sorted(labels_files):
            base = labels_path.replace("_labels.json", "")
            step_path = None
            for ext in [".stp", ".step"]:
                test_path = base + ext
                if os.path.exists(test_path):
                    step_path = test_path
                    break

            if step_path is None:
                ConsoleView.log_warning(
                    f"No matching STEP file found for: {labels_path}"
                )
                continue

            try:
                graph = self._prepare_graph_data(step_path, labels_path)
                graphs.append(graph)
            except Exception as e:
                ConsoleView.log_warning(f"Skipped parsing graph due to: {e}")

        return graphs

    def _run_training_loop(
        self, model: CADFeatureSegmenter, graphs: List[Data], epochs: int
    ) -> None:
        """Executes GNN training epochs and outputs validation metrics reports."""
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.NLLLoss()

        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0
            total_nodes = 0

            for data in graphs:
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = out.argmax(dim=1)
                correct += int((preds == data.y).sum())
                total_nodes += data.num_nodes

            avg_loss = total_loss / len(graphs)
            accuracy = 100.0 * correct / total_nodes
            ConsoleView.draw_progress(epoch, epochs, avg_loss, accuracy)

    def _evaluate_validation(
        self, model: CADFeatureSegmenter, val_graphs: List[Data]
    ) -> None:
        """Runs inference on validation dataset and prints classification report matrix."""
        model.eval()
        y_true_all: List[int] = []
        y_pred_all: List[int] = []

        with torch.no_grad():
            for val_data in val_graphs:
                out = model(val_data)
                preds = out.argmax(dim=1).cpu().numpy()
                y_true_all.extend(val_data.y.cpu().numpy())
                y_pred_all.extend(preds)

        SegmentationMetrics.print_classification_report(
            np.array(y_true_all), np.array(y_pred_all), num_classes=6
        )
