import os
import json
from typing import Dict, List, Tuple

_CONFIG = None


def _load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        path = os.path.join(
            os.path.dirname(__file__), "..", "config", "key_mappings.json"
        )
        with open(path, "r") as f:
            _CONFIG = json.load(f)
    return _CONFIG


def get_class_mapping() -> Tuple[Dict[int, str], List[str]]:
    """Loads and returns the class names and palette lists from config, adjusting dynamically."""
    cfg = _load_config()
    classes = cfg["keyboard_shortcuts"]["classes"]
    colors = cfg["keyboard_shortcuts"].get("class_colors", {})
    categories = {int(k): v for k, v in classes.items()}

    # High-contrast default color set to fall back on if config is missing keys
    default_colors = [
        "#DFE6ED",
        "#00FF00",
        "#FF00FF",
        "#FF9900",
        "#00FFFF",
        "#3366FF",
        "#FF5722",
        "#4CAF50",
        "#9C27B0",
        "#FFEB3B",
        "#00BCD4",
        "#E91E63",
    ]

    palette = []
    for i in range(len(categories)):
        color = colors.get(str(i))
        if color is None:
            color = default_colors[i % len(default_colors)]
        palette.append(color)

    return categories, palette
