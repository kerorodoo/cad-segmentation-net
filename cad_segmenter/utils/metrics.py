import numpy as np
from typing import Dict


class SegmentationMetrics:
    """Computes Confusion Matrix, Precision, Recall, F1-Score, and IoU for CAD face segmentation."""

    @staticmethod
    def compute_confusion_matrix(
        y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6
    ) -> np.ndarray:
        """Computes true vs predicted class count confusion matrix."""
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
        return cm

    @staticmethod
    def print_classification_report(
        y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6
    ) -> None:
        """Prints a formatted Confusion Matrix and classification metrics table."""
        from cad_segmenter.utils.class_config import get_class_mapping

        categories, _ = get_class_mapping()

        cm = SegmentationMetrics.compute_confusion_matrix(y_true, y_pred, num_classes)
        SegmentationMetrics._print_confusion_matrix(cm, categories)
        SegmentationMetrics._print_metrics_table(cm, categories, len(y_true))

    @staticmethod
    def _print_confusion_matrix(cm: np.ndarray, categories: Dict[int, str]) -> None:
        """Helper to print a beautiful, aligned Confusion Matrix."""
        num_classes = min(len(categories), cm.shape[0])
        print("\n" + "=" * 25 + " CONFUSION MATRIX " + "=" * 25)

        header = f"{'True \\ Pred':<12} |"
        for i in range(num_classes):
            header += f" {categories[i]:<30} |"
        print(header)
        print("-" * len(header))

        for i in range(num_classes):
            row = f"{categories[i]:<12} |"
            for j in range(num_classes):
                row += f" {cm[i, j]:<30} |"
            print(row)
        print("=" * len(header))

    @staticmethod
    def _print_metrics_table(
        cm: np.ndarray, categories: Dict[int, str], total_samples: int
    ) -> None:
        """Helper to calculate and print Precision, Recall, F1-Score, and IoU tables."""
        num_classes = min(len(categories), cm.shape[0])
        print("\n" + "=" * 22 + " CLASSIFICATION REPORT " + "=" * 22)
        r_header = f"{'Class':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'IoU':<10} | {'Support':<8}"
        print(r_header)
        print("-" * len(r_header))

        total_correct = 0
        for i in range(num_classes):
            support = int(np.sum(cm[i, :]))
            tp = cm[i, i]
            fp = int(np.sum(cm[:, i])) - tp
            fn = int(np.sum(cm[i, :])) - tp
            total_correct += tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

            print(
                f"{categories[i]:<30} | "
                f"{precision * 100.0:>8.1f}% | "
                f"{recall * 100.0:>8.1f}% | "
                f"{f1 * 100.0:>8.1f}% | "
                f"{iou * 100.0:>8.1f}% | "
                f"{support:<8}"
            )

        print("-" * len(r_header))
        acc = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0
        print(f"Overall Accuracy: {acc:.2f}%")
        print("=" * len(r_header) + "\n")
