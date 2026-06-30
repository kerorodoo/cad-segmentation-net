import os
import json
import numpy as np
import pyvista as pv
from typing import Dict

from cad_segmenter.utils.cad_interactor import CADInteractorStyle
from cad_segmenter.views.console_view import ConsoleView


class AnnotatorViewer3D:
    """Interactive 3D viewport for querying, coloring, and annotating CAD faces."""

    def __init__(self, step_path: str, mesh: pv.PolyData) -> None:
        pv.set_plot_theme("document")
        self.step_path = step_path
        self.mesh = mesh
        self.plotter = pv.Plotter(title="Interactive 3D CAD Annotator Tool")

        # Get total number of distinct faces in the mesh
        self.num_faces = len(np.unique(mesh.cell_data["face_id"]))
        self.face_annotations = np.zeros(self.num_faces, dtype=np.int32)

        # Sync cell data with current face annotations
        self.mesh.cell_data["Annotations"] = self.face_annotations[
            self.mesh.cell_data["face_id"]
        ]
        self.selected_faces = set()

        from cad_segmenter.utils.class_config import get_class_mapping

        self.categories, self.palette = get_class_mapping()

    def show_annotator(self) -> None:
        """Starts the interactive 3D annotation session."""
        self._setup_view()
        self._register_events()
        self.plotter.show()

    def _setup_view(self) -> None:
        """Configures the viewport mesh, color scales, and instruction legend panels."""
        self.plotter.subplot(0, 0)
        self.plotter.add_text("CAD B-Rep Interactive Face Annotator", font_size=12)

        self.plotter.add_mesh(
            self.mesh,
            scalars="Annotations",
            cmap=self.palette,
            clim=[-0.5, len(self.palette) - 0.5],
            show_edges=True,
            edge_color="#37474F",
            name="cad_solid",
            show_scalar_bar=False,
        )

        self._update_selection_text("No face selected. Hover & left-click to pick.")
        self._add_instructions_panel()

    def _register_events(self) -> None:
        """Binds mouse cell picker and keyboard number and saving triggers."""
        # Bind custom CAD interaction and selection callbacks FIRST
        style = CADInteractorStyle()
        style.setup_style(self.plotter, self.mesh, self._on_select)
        self.plotter.iren.style = style

        # Then register key events (binds to the interactor, not the old style)
        for i in self.categories.keys():
            self.plotter.add_key_event(
                str(i), lambda class_id=i: self._assign_class(class_id)
            )

        self.plotter.add_key_event("s", self._save_annotations)
        self.plotter.add_key_event("S", self._save_annotations)

    def _on_select(self, face_idx: int, ctrl_pressed: bool) -> None:
        """Handles face selection and multi-selection toggles."""
        if ctrl_pressed:
            if face_idx in self.selected_faces:
                self.selected_faces.remove(face_idx)
            else:
                self.selected_faces.add(face_idx)
        else:
            self.selected_faces = {face_idx}

        self._update_picked_face_info()
        self._highlight_selected_cells()

    def _assign_class(self, class_id: int) -> None:
        """Assigns class ID to all selected faces and triggers visual refresh."""
        if not self.selected_faces:
            max_class_id = max(self.categories.keys()) if self.categories else 5
            self._update_selection_text(
                f"⚠️ ERROR: Select face(s) first before pressing 0-{max_class_id}!"
            )
            return

        # Bulk paint all currently selected faces
        for face_idx in self.selected_faces:
            self.face_annotations[face_idx] = class_id

        # Update cell scalars and recolor mesh instantly
        cell_preds = self.face_annotations[self.mesh.cell_data["face_id"]]
        self.mesh.cell_data["Annotations"] = cell_preds

        self._update_picked_face_info()
        self._highlight_selected_cells()  # Maintain highlighted boundaries

    def _update_picked_face_info(self) -> None:
        """Queries and displays live status card of selected face(s)."""
        max_class_id = max(self.categories.keys()) if self.categories else 5
        if len(self.selected_faces) == 1:
            idx = list(self.selected_faces)[0]
            cls_id = self.face_annotations[idx]
            cls_name = self.categories.get(cls_id, "Unassigned")
            info = (
                f"Active Selection:\n"
                f"-----------------\n"
                f"Face ID: #{idx:02d}\n"
                f"Current Class: {cls_name}\n"
                f"👉 Press 0-{max_class_id} to change category.\n"
                f"👉 Press 'S' to save annotations!"
            )
        elif len(self.selected_faces) > 1:
            info = (
                f"Active Selections:\n"
                f"-----------------\n"
                f"Total Selected: {len(self.selected_faces)} faces\n"
                f"IDs: {sorted(list(self.selected_faces))}\n"
                f"👉 Press 0-{max_class_id} to bulk paint selection.\n"
                f"👉 Press 'S' to save annotations!"
            )
        else:
            info = "Active Selection:\n-----------------\nNo face selected."

        self._update_selection_text(info)
        # Simultaneously log the selection details to the console terminal
        ConsoleView.log_info(info.replace("\n", " | "))

    def _highlight_selected_cells(self) -> None:
        """Extracts and overlays a semi-transparent yellow highlight on selected faces."""
        if self.selected_faces:
            cell_face_ids = self.mesh.cell_data["face_id"]
            cell_mask = np.isin(cell_face_ids, list(self.selected_faces))
            selected_indices = np.where(cell_mask)[0]

            if len(selected_indices) > 0:
                highlight_mesh = self.mesh.extract_cells(selected_indices)
                actor = self.plotter.add_mesh(
                    highlight_mesh,
                    color="yellow",
                    opacity=0.4,
                    name="selection_highlight",
                    show_edges=True,
                    edge_color="yellow",
                )
                actor.SetPickable(False)
        else:
            self.plotter.remove_actor("selection_highlight")

    def _save_annotations(self) -> None:
        """Writes current annotations to JSON file format in data/."""
        basename = os.path.splitext(os.path.basename(self.step_path))[0]
        os.makedirs("data", exist_ok=True)
        out_path = os.path.join("data", f"{basename}_labels.json")

        manifest: Dict[str, int] = {}
        for idx, cls_id in enumerate(self.face_annotations):
            manifest[str(idx)] = int(cls_id)

        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)

        save_notice = (
            f"SUCCESS:\n"
            f"-----------------\n"
            f"Serialized {len(manifest)} faces successfully!\n"
            f"Output written to:\n"
            f"{out_path}"
        )
        self.plotter.add_text(
            save_notice,
            position="upper_right",
            font_size=10,
            color="darkgreen",
            name="save_info",
        )

    def _update_selection_text(self, text: str) -> None:
        """Draws or updates picked face text overlay card in upper-left."""
        self.plotter.add_text(
            text,
            position="upper_left",
            font_size=10,
            color="darkred",
            name="selection_info",
        )

    def _add_instructions_panel(self) -> None:
        """Draws the persistent keyboard shortcut keys instructions on screen."""
        shortcut_lines = ["Keyboard Annotator shortcuts:"]
        for idx in sorted(self.categories.keys()):
            name = self.categories[idx]
            shortcut_lines.append(f"{idx}: {name}")
        shortcut_lines.append("S: Save annotations to data/\n")
        shortcut_lines.extend(
            [
                "🖱️ Shift+Drag: Pan model",
                "🖱️ Alt+Drag: Rotate model",
                "🖱️ Scroll Wheel: Zoom/Scale",
                "🖱️ Left-Click: Select face",
                "🖱️ Ctrl+Left-Click: Multi-Select",
            ]
        )
        shortcuts = "\n".join(shortcut_lines)
        self.plotter.add_text(
            shortcuts,
            position="lower_left",
            font_size=9,
            color="black",
            name="annotator_legend",
        )
