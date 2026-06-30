import os
import json
import numpy as np
import pyvista as pv
from torch_geometric.data import Data

from cad_segmenter.utils.cad_interactor import CADInteractorStyle
from cad_segmenter.views.console_view import ConsoleView


class CADViewer3D:
    """Side-by-side 3D interactive engineering inspection dashboard in PyVista."""

    def __init__(self) -> None:
        pv.set_plot_theme("document")
        self.plotter = pv.Plotter(shape=(1, 3), title="CAD Segmentation Net HUD")
        self.selected_faces = set()
        self._load_config()
        from cad_segmenter.utils.class_config import get_class_mapping

        self.categories, self.palette = get_class_mapping()

    def _load_config(self) -> None:
        """Loads keyboard/mouse configurations from central JSON file."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "key_mappings.json",
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def show_inspection(
        self,
        mesh: pv.PolyData,
        face_preds: np.ndarray,
        face_probs: np.ndarray,
        graph: Data,
        warnings: list = None,
        face_labels: np.ndarray = None,
    ) -> None:
        """Configures side-by-side subplots and shows the dual-pane viewport."""
        self.mesh = mesh
        self.face_preds = face_preds
        self.face_probs = face_probs
        self.graph = graph
        self.face_labels = face_labels

        cell_face_ids = mesh.cell_data["face_id"]
        mesh.cell_data["Predictions"] = face_preds[cell_face_ids]

        if face_labels is not None:
            mesh.cell_data["Ground_Truth"] = face_labels[cell_face_ids]

        # Map normals to cells
        face_normals_z = graph.x[:, 4].cpu().numpy()
        mesh.cell_data["Surface_Gradient"] = face_normals_z[cell_face_ids]

        # Setup Panes
        self._setup_left_pane(mesh)
        self._setup_fag_pane(mesh, graph)
        self._setup_right_pane(mesh, warnings)

        self.plotter.link_views()
        self.plotter.show()

    def _setup_left_pane(self, mesh: pv.PolyData) -> None:
        """Sets up the left subplot for input physical metrics or ground truth labels if available."""
        self.plotter.subplot(0, 0)
        if self.face_labels is not None:
            self.plotter.add_text("Ground Truth Annotation Labels", font_size=10)
            self.plotter.add_mesh(
                mesh,
                scalars="Ground_Truth",
                cmap=self.palette,
                clim=[-0.5, len(self.palette) - 0.5],
                show_edges=True,
                edge_color="#37474F",
                show_scalar_bar=False,
            )
            self._add_class_legend()
        else:
            self.plotter.add_text(
                "Input Metrics: Surface Normal (Z-Axis Orientation)", font_size=10
            )
            self.plotter.add_mesh(
                mesh,
                scalars="Surface_Gradient",
                cmap="coolwarm",
                show_edges=True,
                edge_color="#37474F",
            )

    def _setup_right_pane(self, mesh: pv.PolyData, warnings: list = None) -> None:
        """Sets up the right subplot for GNN prediction categories & DFM highlights."""
        self.plotter.subplot(0, 2)
        self.plotter.add_text("GNN Semantic Segmentation Output", font_size=10)

        self.plotter.add_mesh(
            mesh,
            scalars="Predictions",
            cmap=self.palette,
            clim=[-0.5, len(self.palette) - 0.5],
            show_edges=True,
            edge_color="#37474F",
            name="cad_solid",
            show_scalar_bar=False,
        )

        self._add_class_legend()
        self._update_hud_selection_overlays()

        # Bind custom CAD interaction and selection callbacks
        style = CADInteractorStyle()
        style.setup_style(self.plotter, mesh, self._on_select)
        self.plotter.iren.style = style

        # Highlight DFM Warnings
        if warnings:
            for warn in warnings:
                self.plotter.add_point_labels(
                    [warn["pos"]],
                    [warn["msg"]],
                    point_color="red",
                    text_color="red",
                    font_size=10,
                    shape_color="yellow",
                )

    def _on_select(self, face_idx: int, ctrl_pressed: bool) -> None:
        """Handles selection/multi-selection and updates highlighting."""
        if ctrl_pressed:
            if face_idx in self.selected_faces:
                self.selected_faces.remove(face_idx)
            else:
                self.selected_faces.add(face_idx)
        else:
            self.selected_faces = {face_idx}

        self._update_hud_selection_overlays()

    def _update_hud_selection_overlays(self) -> None:
        """Updates the highlighted selected face meshes and on-screen card text."""
        if len(self.selected_faces) == 1:
            idx = list(self.selected_faces)[0]
            cls_name = self.categories.get(self.face_preds[idx], "Unassigned")
            prob = self.face_probs[idx] * 100.0
            info = f"Selected Face Query:\n-----------------\nFace ID: #{idx:02d}\nClass: {cls_name}\nConfidence: {prob:.1f}%"
            if self.face_labels is not None:
                gt_class_id = self.face_labels[idx]
                gt_name = self.categories.get(gt_class_id, "Unassigned")
                info += f"\nGround Truth: {gt_name}"
        elif len(self.selected_faces) > 1:
            info = f"Selected Faces Query:\n-----------------\nTotal Selected: {len(self.selected_faces)} faces\nIDs: {sorted(list(self.selected_faces))}"
        else:
            info = "Selected Face Query:\n-----------------\nNo face selected."

        self.plotter.add_text(
            info, position="upper_left", font_size=10, color="darkred", name="pick_info"
        )
        # Simultaneously log face selection to the console
        ConsoleView.log_info(info.replace("\n", " | "))
        self._highlight_selected_cells()

    def _highlight_selected_cells(self) -> None:
        """Extracts and overlays a semi-transparent yellow highlight on selected faces on both subplots."""
        if self.selected_faces:
            cell_face_ids = self.mesh.cell_data["face_id"]
            cell_mask = np.isin(cell_face_ids, list(self.selected_faces))
            selected_indices = np.where(cell_mask)[0]

            if len(selected_indices) > 0:
                highlight_mesh = self.mesh.extract_cells(selected_indices)
                self.plotter.subplot(0, 0)
                actor0 = self.plotter.add_mesh(
                    highlight_mesh,
                    color="yellow",
                    opacity=0.4,
                    name="selection_highlight",
                    show_edges=True,
                    edge_color="yellow",
                )
                actor0.SetPickable(False)
                self.plotter.subplot(0, 2)
                actor1 = self.plotter.add_mesh(
                    highlight_mesh,
                    color="yellow",
                    opacity=0.4,
                    name="selection_highlight",
                    show_edges=True,
                    edge_color="yellow",
                )
                actor1.SetPickable(False)
        else:
            self.plotter.subplot(0, 0)
            self.plotter.remove_actor("selection_highlight")
            self.plotter.subplot(0, 2)
            self.plotter.remove_actor("selection_highlight")

    def _add_class_legend(self) -> None:
        """Adds a beautiful visual legend with colored swatches and class mappings."""
        entries = []
        for idx in range(len(self.palette)):
            name = self.categories[idx]
            entries.append([f"{idx}: {name}", self.palette[idx], "rectangle"])
        self.plotter.add_legend(
            entries,
            bcolor="white",
            background_opacity=0.85,
            border=True,
            size=(0.40, 0.30),
            face="none",
            loc="upper right",
            name="class_legend",
        )
        interaction_text = (
            "🖱️ Shift+Drag: Pan model\n"
            "🖱️ Alt+Drag: Rotate model\n"
            "🖱️ Scroll Wheel: Zoom/Scale\n"
            "🖱️ Left-Click: Select face\n"
            "🖱️ Ctrl+Left-Click: Multi-Select"
        )
        self.plotter.add_text(
            interaction_text,
            position="lower_left",
            font_size=9,
            color="black",
            name="mouse_hints",
        )

    def _setup_fag_pane(self, mesh: pv.PolyData, graph: Data) -> None:
        """Sets up the middle subplot for the Face Adjacency Graph (FAG) visualization."""
        self.plotter.subplot(0, 1)
        self.plotter.add_text("Face Adjacency Graph (FAG)", font_size=10)
        self.plotter.add_text(
            "👉 Press 'F' to toggle FAG elements",
            position="lower_left",
            font_size=9,
            color="black",
            name="fag_toggle_hint",
        )

        # Translucent base mesh for contextual outline
        self.plotter.add_mesh(
            mesh,
            color="#E0E0E0",
            opacity=0.25,
            show_edges=True,
            edge_color="#B0BEC5",
            name="fag_base_mesh",
            pickable=False,
        )

        centroids = np.array(graph.centroids)

        # Nodes (Centroids)
        nodes_mesh = pv.PolyData(centroids)
        self.fag_nodes_actor = self.plotter.add_mesh(
            nodes_mesh,
            color="#3F51B5",  # Royal Indigo/Blue
            render_points_as_spheres=True,
            point_size=12,
            name="fag_nodes",
            pickable=False,
        )

        # Edges (Adjacency connections)
        edge_index = graph.edge_index.cpu().numpy()
        mask = edge_index[0] < edge_index[1]
        filtered_edges = edge_index[:, mask]

        num_edges = filtered_edges.shape[1]
        lines = np.empty(3 * num_edges, dtype=np.int32)
        lines[0::3] = 2
        lines[1::3] = filtered_edges[0]
        lines[2::3] = filtered_edges[1]

        edges_mesh = pv.PolyData(centroids)
        edges_mesh.lines = lines

        self.fag_edges_actor = self.plotter.add_mesh(
            edges_mesh,
            color="#E91E63",  # Vibrant Pink/Red for graph edges
            line_width=2,
            name="fag_edges",
            pickable=False,
        )

        # Default: FAG elements visible by default, toggleable with 'F'
        self.fag_nodes_actor.SetVisibility(True)
        self.fag_edges_actor.SetVisibility(True)

        # Key binds for toggling visibility
        self.plotter.add_key_event("f", self._toggle_fag_visibility)
        self.plotter.add_key_event("F", self._toggle_fag_visibility)

    def _toggle_fag_visibility(self) -> None:
        """Toggles visibility of the FAG nodes and edges."""
        if hasattr(self, "fag_nodes_actor") and hasattr(self, "fag_edges_actor"):
            current_vis = self.fag_nodes_actor.GetVisibility()
            new_vis = not current_vis
            self.fag_nodes_actor.SetVisibility(new_vis)
            self.fag_edges_actor.SetVisibility(new_vis)
            status = "Visible" if new_vis else "Hidden"
            ConsoleView.log_info(f"FAG Overlay visibility toggled to: {status}")
            self.plotter.render()
