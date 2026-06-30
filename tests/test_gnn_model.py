import os
import shutil
import torch
from cad_segmenter.models.data_factory import StructuralDataFactory
from cad_segmenter.models.cad_graph import CADGraphModel
from cad_segmenter.models.gnn_model import CADFeatureSegmenter


def test_gnn_forward() -> None:
    """Verifies that the GNN forward pass produces correct tensor output shape."""
    test_dir = "data/test_synthetic_gnn"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    factory = StructuralDataFactory(test_dir)
    step_path, _ = factory.build_variant(serial_id=1)

    model = CADGraphModel(step_path)
    graph = model.extract_graph_tensors()

    segmenter = CADFeatureSegmenter(in_channels=6, num_classes=4)
    segmenter.eval()

    with torch.no_grad():
        out = segmenter(graph)

    assert out is not None
    assert out.size(0) == graph.x.size(0)
    assert out.size(1) == 4  # 4 classification logits

    # Cleanup test output
    shutil.rmtree(test_dir)
