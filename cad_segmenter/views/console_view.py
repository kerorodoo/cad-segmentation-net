import sys


class ConsoleView:
    """Handles professional color-coded logging and progress outputs to the terminal."""

    # ANSI escape sequences for text styling
    HEADER = "\033[95m"
    INFO = "\033[94m"
    SUCCESS = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

    @staticmethod
    def log_header(msg: str) -> None:
        """Logs an styled, highlighted title header."""
        print(
            f"\n{ConsoleView.BOLD}{ConsoleView.HEADER}=== {msg} ==={ConsoleView.ENDC}"
        )

    @staticmethod
    def log_info(msg: str) -> None:
        """Logs an informational message."""
        print(f"[{ConsoleView.INFO}INFO{ConsoleView.ENDC}] {msg}")

    @staticmethod
    def log_warning(msg: str) -> None:
        """Logs a warning notification."""
        print(f"[{ConsoleView.WARNING}WARNING{ConsoleView.ENDC}] {msg}")

    @staticmethod
    def log_success(msg: str) -> None:
        """Logs a success confirmation."""
        print(f"[{ConsoleView.SUCCESS}SUCCESS{ConsoleView.ENDC}] {msg}")

    @staticmethod
    def log_error(msg: str) -> None:
        """Logs an error trace."""
        print(f"[{ConsoleView.FAIL}ERROR{ConsoleView.ENDC}] {msg}", file=sys.stderr)

    @staticmethod
    def draw_progress(epoch: int, total_epochs: int, loss: float, acc: float) -> None:
        """Renders an interactive ASCII training progress bar."""
        percent = int(100 * (epoch / total_epochs))
        bar_length = 20
        filled_length = int(bar_length * epoch // total_epochs)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        sys.stdout.write(
            f"\rEpoch {epoch:02d}/{total_epochs:02d} |{bar}| "
            f"{percent}% | Loss: {loss:.4f} | Acc: {acc:.2f}%"
        )
        sys.stdout.flush()
        if epoch == total_epochs:
            print()  # Final newline
