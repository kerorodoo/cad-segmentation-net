import numpy as np
from cad_segmenter.utils.metrics import SegmentationMetrics


def test_metrics_report() -> None:
    """Verifies that SegmentationMetrics prints reports and handles accuracies correctly."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2])
    y_pred = np.array(
        [0, 1, 2, 3, 4, 5, 0, 1, 0]
    )  # One mistake: true class 2 predicted as class 0

    # Ensure running report calculation and output is successful
    SegmentationMetrics.print_classification_report(y_true, y_pred, num_classes=6)

    cm = SegmentationMetrics.compute_confusion_matrix(y_true, y_pred, num_classes=6)
    assert cm.shape == (6, 6)
    assert cm[0, 0] == 2  # True class 0, pred class 0 count
    assert cm[2, 0] == 1  # True class 2, pred class 0 count (mistake)
