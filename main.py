import argparse
import sys

from cad_segmenter.controllers.app_controller import AppController
from cad_segmenter.controllers.train_controller import TrainController
from cad_segmenter.views.console_view import ConsoleView


def main() -> None:
    """Main CLI entry point to run different pipeline modes."""
    args = _parse_args()

    try:
        if args.generate:
            controller = TrainController()
            controller._generate_dataset(num_variants=args.num_variants)
            ConsoleView.log_success("Dataset generation task finished.")

        elif args.bootstrap:
            controller = TrainController()
            controller.bootstrap_and_train(
                num_variants=args.num_variants,
                epochs=args.epochs,
                use_existing_dataset=args.use_existing_dataset,
            )
            ConsoleView.log_success("Model self-bootstrapping complete!")

        elif args.predict:
            controller = AppController()
            controller.predict_and_visualize(
                step_path=args.predict, weights_path=args.weights
            )

        elif args.annotate:
            controller = AppController()
            controller.annotate_step(step_path=args.annotate)

        else:
            ConsoleView.log_info(
                "No action specified. Run with `--help` to see available modes."
            )

    except Exception as e:
        ConsoleView.log_error(f"Execution failed: {e}")
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CAD Segmentation Net - End-to-End B-Rep to GNN Classifier"
    )
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic training examples only.",
    )
    group.add_argument(
        "--bootstrap",
        action="store_true",
        help="Generate synthetic data and train the GNN to produce pre-trained weights.",
    )
    group.add_argument(
        "--predict",
        type=str,
        metavar="FILE.stp",
        help="Segment a custom STEP file and show interactive dual-pane HUD.",
    )
    group.add_argument(
        "--annotate",
        type=str,
        metavar="FILE.stp",
        help="Launch the interactive 3D face annotator and painter session.",
    )

    parser.add_argument(
        "--num-variants",
        type=int,
        default=15,
        help="Number of synthetic variants to generate (default: 15).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of GNN training epochs (default: 25).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to custom model weights .pth (optional).",
    )
    parser.add_argument(
        "--use-existing-dataset",
        action="store_true",
        help="Use already prepared datasets in data/train and data/val for training.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
