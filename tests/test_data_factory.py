import os
import shutil
from cad_segmenter.models.data_factory import StructuralDataFactory


def test_data_factory() -> None:
    """Verifies that StructuralDataFactory generates STEP and labels files."""
    test_dir = "data/test_synthetic"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    factory = StructuralDataFactory(test_dir)
    step_path, labels_path = factory.build_variant(serial_id=1)

    assert os.path.exists(step_path)
    assert os.path.exists(labels_path)
    assert step_path.endswith(".stp")
    assert labels_path.endswith(".json")

    # Cleanup test output
    shutil.rmtree(test_dir)
