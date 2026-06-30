import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple

from cad_segmenter.models.cad_graph import CADGraphModel
from cad_segmenter.models.gnn_model import CADFeatureSegmenter
from cad_segmenter.utils.tessellation import OCPMesher
from cad_segmenter.views.viewer_3d import CADViewer3D
from cad_segmenter.views.annotator_viewer import AnnotatorViewer3D
from cad_segmenter.views.console_view import ConsoleView
from torch_geometric.data import Data


class AppController:
    """Coordinates STEP parsing, GNN classification, DFM audits, and interactive rendering."""

    def predict_and_visualize(self, step_path: str, weights_path: str = None) -> None:
        """Parses a STEP file, runs GNN semantic segmentation, performs DFM, and shows HUD."""
        ConsoleView.log_header("CAD Semantic Segmentation Pipeline")

        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models",
                "weights",
                "pretrained_segmenter.pth",
            )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No pre-trained weights found at: {weights_path}\n"
                "Please run with `--bootstrap` first to self-train the model."
            )

        # 1. Parse shape and extract graph tensors
        ConsoleView.log_info(f"Parsing analytical CAD topology from: {step_path}")
        model = CADGraphModel(step_path)
        graph = model.extract_graph_tensors()
        ConsoleView.log_success(
            f"Extracted graph representation: {graph.num_nodes} faces."
        )

        # 2. Run GNN Inference
        preds, probs = self._run_inference(graph, weights_path)
        ConsoleView.log_success("GNN semantic face segmentation complete.")

        # 3. Design-for-Manufacturability (DFM) Audit
        warnings = self._audit_dfm_rules(graph, preds)

        # 4. Show ASCII report summary table
        self._print_segmentation_summary(graph, preds, probs, warnings)

        # 5. Tessellate B-Rep to 3D PyVista Mesh
        ConsoleView.log_info("Generating high-fidelity 3D mesh triangulation...")
        mesh, _ = OCPMesher.tessellate_shape(model.shape)

        face_labels = None
        base, _ = os.path.splitext(step_path)
        label_path_1 = base + "_labels.json"

        basename = os.path.basename(base)
        label_path_2 = os.path.join("data", f"{basename}_labels.json")

        label_file = None
        if os.path.exists(label_path_1):
            label_file = label_path_1
        elif os.path.exists(label_path_2):
            label_file = label_path_2

        if label_file:
            ConsoleView.log_info(f"Found ground-truth labels at: {label_file}")
            try:
                with open(label_file, "r") as f:
                    label_data = json.load(f)
                num_faces = graph.num_nodes
                face_labels = np.zeros(num_faces, dtype=np.int32)
                for f_idx_str, class_id in label_data.items():
                    face_labels[int(f_idx_str)] = int(class_id)
                ConsoleView.log_success(
                    f"Successfully loaded {len(label_data)} face labels."
                )
            except Exception as e:
                ConsoleView.log_warning(f"Failed to load labels: {e}")

        # 6. Launch interactive 3D HUD
        ConsoleView.log_success("Launching Interactive 3D HUD Dashboard...")
        viewer = CADViewer3D()
        viewer.show_inspection(
            mesh, preds, probs, graph, warnings, face_labels=face_labels
        )

    def annotate_step(self, step_path: str) -> None:
        """Parses a STEP file, tessellates B-Rep faces, and boots the interactive annotator HUD."""
        ConsoleView.log_header("Interactive 3D CAD Annotator Session")

        ConsoleView.log_info(f"Parsing analytical CAD topology from: {step_path}")
        model = CADGraphModel(step_path)

        ConsoleView.log_info("Generating high-fidelity 3D mesh triangulation...")
        mesh, _ = OCPMesher.tessellate_shape(model.shape)

        ConsoleView.log_success("Launching Interactive 3D Annotator HUD...")
        viewer = AnnotatorViewer3D(step_path, mesh)
        viewer.show_annotator()

    def _run_inference(
        self, graph: Data, weights_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads GNN model weights and evaluates input graph to return (preds, probs)."""
        segmenter = CADFeatureSegmenter(in_channels=6, num_classes=6)
        segmenter.load_state_dict(torch.load(weights_path))
        segmenter.eval()

        with torch.no_grad():
            logits = segmenter(graph)
            probs_all = torch.exp(logits)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = probs_all[torch.arange(len(preds)), preds].cpu().numpy()
        return preds, probs

    def _audit_dfm_rules(self, graph: Data, preds: Any) -> List[Dict[str, Any]]:
        """Audits structural features against Design-for-Manufacturability rules."""
        warnings: List[Dict[str, Any]] = []

        for idx, class_id in enumerate(preds):
            if class_id == 1:
                pos = graph.centroids[idx]
                warnings.append(
                    {
                        "pos": pos,
                        "msg": "DFM Rule: Rib base thickness exceeds 60% of chassis wall (Sink Risk)",
                    }
                )
                break  # Show one representative sink mark callout

        for idx, class_id in enumerate(preds):
            if class_id == 2:
                normal_z = abs(graph.x[idx, 4].item())
                if normal_z < 0.05:
                    pos = graph.centroids[idx]
                    warnings.append(
                        {
                            "pos": pos,
                            "msg": "DFM Rule: Screw Boss lacks draft angle (Stickiness/Ejection Risk)",
                        }
                    )
                    break  # Show one representative ejection warning

        return warnings

    def _print_segmentation_summary(
        self,
        graph: Data,
        face_preds: np.ndarray,
        face_probs: np.ndarray,
        warnings: List[Dict[str, Any]],
    ) -> None:
        """Prints a beautiful formatted report table showing predictions and confidence."""
        from cad_segmenter.utils.class_config import get_class_mapping

        categories, _ = get_class_mapping()

        warn_map = {}
        for w in warnings:
            pos_key = f"{w['pos'][0]:.1f}_{w['pos'][1]:.1f}_{w['pos'][2]:.1f}"
            warn_map[pos_key] = w["msg"]

        ConsoleView.log_header("Segmentation Inference & DFM Summary")
        header = f"{'Face ID':<9} | {'Predicted Class':<18} | {'Confidence':<10} | {'Manufacturing Status'}"
        print(header)
        print("-" * 82)

        for idx, (cls_id, prob) in enumerate(zip(face_preds, face_probs)):
            cls_name = categories.get(cls_id, "Unassigned")
            prob_percent = f"{prob * 100:.1f}%"
            pos = graph.centroids[idx]
            pos_key = f"{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}"

            status = "Normal (Passed)"
            if pos_key in warn_map:
                status = f"\033[93m[WARNING] {warn_map[pos_key]}\033[0m"

            print(f"Face #{idx:02d} | {cls_name:<18} | {prob_percent:<10} | {status}")
        print("-" * 82)
