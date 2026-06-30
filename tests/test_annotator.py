import os
import json
import numpy as np
import pyvista as pv
from cad_segmenter.views.annotator_viewer import AnnotatorViewer3D


def test_annotator_serialization() -> None:
    """Verifies that AnnotatorViewer3D correctly saves the face annotations dictionary."""
    mesh = pv.Sphere(radius=1.0)
    # Mock face_id cell mapping
    mesh.cell_data["face_id"] = np.zeros(mesh.n_cells, dtype=np.int32)

    step_path = "dummy.stp"
    viewer = AnnotatorViewer3D(step_path, mesh)

    # Assign class 3 to face 0
    viewer.face_annotations[0] = 3

    # Trigger serialization
    viewer._save_annotations()

    out_path = os.path.join("data", "dummy_labels.json")
    assert os.path.exists(out_path)

    with open(out_path, "r") as f:
        manifest = json.load(f)

    assert manifest["0"] == 3

    # Cleanup
    os.remove(out_path)
