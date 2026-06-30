import os
import shutil
from cad_segmenter.models.data_factory import StructuralDataFactory
from cad_segmenter.models.cad_graph import CADGraphModel


def test_cad_graph_extraction() -> None:
    """Verifies that CADGraphModel parses face-level adjacency graphs correctly."""
    test_dir = "data/test_synthetic_graph"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    factory = StructuralDataFactory(test_dir)
    step_path, _ = factory.build_variant(serial_id=1)

    model = CADGraphModel(step_path)
    graph = model.extract_graph_tensors()

    assert graph is not None
    assert graph.x is not None
    assert graph.x.dim() == 2
    assert (
        graph.x.size(1) == 6
    )  # 6 node features: type, area, normal_x, normal_y, normal_z, centroid_z
    assert len(graph.centroids) == graph.x.size(0)
    assert len(graph.face_id_map) == graph.x.size(0)

    # Cleanup test output
    shutil.rmtree(test_dir)
