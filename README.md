# CAD Segmentation Net: End-to-End B-Rep to GNN Feature Segmentation

An enterprise-grade, object-oriented (OOP) machine learning application that parses **3D CAD STEP files (.stp)** directly into **Graph Neural Network (GNN) tensors**, performs semantic face-level segmentation (e.g., classifying Chassis Base, Stiffener Ribs, and Screw Bosses), conducts automated **Design-for-Manufacturability (DFM) physical rule audits**, and renders results in a high-fidelity, side-by-side **Dual-Pane interactive 3D HUD** using PyVista.

Designed and implemented in Python under strict **Model-View-Controller (MVC)** architectural separation and strong type hints (`typing`), keeping functions modular and under 30 lines.

---

## 🏗️ System Architecture & Data Flow

```
+---------------------------------------------------------------------------------+
|                                    DATA FACTORY                                 |
|  [Randomized Params] ───► [CadQuery Solids] ───► [.stp Part] & [.json Labels]  |
+------------------------------------------------------+--------------------------+
                                                       │
                                                       ▼
+---------------------------------------------------------------------------------+
|                                 MODELS LAYER                                    |
|  [STEP File] ───► [OCP B-Rep Explorer] ───► [PyG Data Graph: Node & Edge Tensors]|
+------------------------------------------------------+--------------------------+
                                                       │
                                                       ▼
+---------------------------------------------------------------------------------+
|                              CONTROLLERS & INFERENCE                            |
|  [PyTorch Geometric GNN] ───► [Class Probability Logits] ───► [DFM Rule Audits] |
+------------------------------------------------------+--------------------------+
                                                       │
                                                       ▼
+---------------------------------------------------------------------------------+
|                                PRESENTATION LAYER                               |
|  [Custom Tessellation] ───► [Dual-Pane PyVista HUD: Normal Scales vs. Semantics]|
+---------------------------------------------------------------------------------+
```

---

## 🌟 Key Features

1.  **Analytical B-Rep Graph Extraction (`models/cad_graph.py`)**:
    Traverses STEP files directly using low-level OpenCascade (`OCP`) bindings. Extracts topological relationships, assigning faces as graph nodes with 6-dimensional feature vectors $[SurfaceType, Area, Normal\_X, Normal\_Y, Normal\_Z, Centroid\_Z]$ and shared edges as graph edges with dihedral-angle attributes. Supports 6 distinct component classes: `plate`, `rib`, `column`, `clip(hook)`, `hole`, and `zifu`.
2.  **High-Fidelity Tessellation Mapper (`utils/tessellation.py`)**:
    Uses OpenCascade's triangulation generator to tessellate actual CAD faces and preserves face order. Triangles in the resulting PyVista `PolyData` mesh are assigned a `"face_id"` matching the node graph index. This avoids simple proxy shapes (like spheres) and renders the **actual 3D geometry** fully segmented in the viewer!
3.  **Geometric GNN Classifier (`models/gnn_model.py`)**:
    A PyG-based deep graph neural network (`CADFeatureSegmenter`) employing Graph Convolutional layers, dropout, and a 6-class classification head.
4.  **Automatic Design-for-Manufacturability (DFM) Audits (`controllers/app_controller.py`)**:
    Combines neural classification labels with local geometry to execute physical rules:
    *   **Sink Mark Risk**: Raises warnings if a face classified as a `rib` exceeds 60% of the nominal base-plate wall thickness.
    *   **Boss Draft Angle**: Warns if a cylindrical `column` lacks a draft angle, creating ejection risks in injection molding.
5.  **Interactive 3D Dual-Pane HUD (`views/viewer_3d.py`)**:
    Renders the physical slope metrics (normals scale) in the left viewport and GNN classifications/DFM flags as 3D text cards with confidence percentages overlaid directly over face centroids. Supports linked, synchronized camera viewpoints.
6.  **Interactive CLI progress & Evaluation Report (`utils/metrics.py`)**:
    Generates a pure-Python validation **Confusion Matrix** and **Classification Report** (Precision, Recall, F1, IoU per class, and Overall Accuracy) printed beautifully in the terminal at the end of training. Handles progress bars during epoch runs.

---

## 📁 Directory Structure

```text
cad-segmentation-net/
│
├── main.py                     # CLI Entry Point
│
├── cad_segmenter/              # Core Application Package
│   ├── __init__.py
│   │
│   ├── models/                 # DATA & SCHEMA LAYER (State and Parsers)
│   │   ├── __init__.py
│   │   ├── cad_graph.py        # OCP topological graph extractor & PyG graph builder
│   │   ├── gnn_model.py        # PyTorch Geometric GNN architecture definition
│   │   ├── data_factory.py     # Synthetic procedural generator (chassis base/bosses/ribs)
│   │   └── weights/
│   │       └── .gitkeep        # Directory for serialized pre-trained weights (.pth)
│   │
│   ├── controllers/            # BUSINESS LOGIC LAYER (Workflows & Physics Audits)
│   │   ├── __init__.py
│   │   ├── app_controller.py   # Coordinates loader -> GNN -> DFM audits -> Viewer
│   │   └── train_controller.py # Coordinates Generator -> GNN Trainer -> Weights Saver
│   │
│   ├── views/                  # PRESENTATION LAYER (Terminals & 3D HUD Viewports)
│   │   ├── __init__.py
│   │   ├── viewer_3d.py        # Synchronized Dual-Pane PyVista HUD Viewer
│   │   └── console_view.py     # ANSI styled CLI logging and training progress indicators
│   │
│   └── utils/                  # CORE UTILITIES
│       ├── __init__.py
│       └── tessellation.py     # Analytical OCP shape to PyVista triangulation converter
│
└── tests/                      # AUTOMATED VERIFICATION SUITE
    ├── __init__.py
    ├── test_cad_graph.py       # Verifies OCP graph extraction structures
    ├── test_gnn_model.py       # Verifies neural dimensions and forward passes
    └── test_data_factory.py    # Verifies CadQuery procedural shapes creation
```

---

## 🛠️ Installation & Setup

Because compiling heavy libraries (like PyTorch, PyG, and OpenCascade) can take considerable storage space, and root partitions are often limited, we use a custom virtual environment setup pointing entirely to the spacious `/mnt/data800g` drive.

### 1. Initialize Virtual Environment (PEP 668 compliant)
We create a local isolated environment without the default pip wrapper and manually bootstrap `get-pip` directly within `/mnt/data800g`:
```bash
# Create venv structure
python3 -m venv venv --without-pip

# Download and install local PIP
python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
./venv/bin/python3 get-pip.py
rm get-pip.py
```

### 2. Install Dependencies
Configure pip temporary and cache folders to utilize `/mnt/data800g` to bypass any root `/` partition limits:
```bash
# Create custom temporary and cache folders
mkdir -p /mnt/data800g/chin/.tmp
mkdir -p /mnt/data800g/chin/.cache/pip

# Run local installation
export TMPDIR=/mnt/data800g/chin/.tmp
export PIP_CACHE_DIR=/mnt/data800g/chin/.cache/pip
./venv/bin/pip install torch torchvision torch-geometric pyvista trimesh ruff pytest cadquery
```

---

## 🚀 Execution & Command-Line Usage

The application provides three clean mutual-exclusive CLI modes:

### 1. Bootstrap / Model Self-Training (Recommended)
This mode procedurally generates high-quality synthetic CAD variations with perfect labels, divides them into Train (`data/train`), Validation (`data/val`), and Test (`data/synthetic`) directories, trains the GNN, evaluates it against the validation set to print a **Confusion Matrix Classification Report**, and serializes the final model weights.

Each training execution dynamically initializes a dedicated, timestamped folder at the project root named `task_YYYYMMDD_HHMMSS/` containing:
* **`weights/`**:
  * `best_segmenter.pth`: Weights from the epoch with the lowest validation loss (also mirrors to `pretrained_segmenter.pth` and the default global `cad_segmenter/models/weights/pretrained_segmenter.pth` folder for seamless inference).
  * `latest_segmenter.pth`: Weights at the final epoch of training.
* **`charts/`**:
  * `loss_curves.png`: Loss curves (train vs. val) for each epoch.
  * `accuracy_curves.png`: Accuracy curves (train vs. val) for each epoch.
* **`training.log`**: A complete text recording of every metric, message, progress bar, and evaluation table printed to the console during the training run.

```bash
./venv/bin/python3 main.py --bootstrap --num-variants 15 --epochs 25
```

Alternatively, to train the model using an already prepared dataset residing in `data/train` and `data/val` (without generating new synthetic files), add the `--use-existing-dataset` flag:
```bash
./venv/bin/python3 main.py --bootstrap --use-existing-dataset --epochs 25
```

You can also customize the training process with standard optimization parameters:
* **Custom Learning Rate:** Use `--lr <value>` or `--learning-rate <value>` (default: `0.01`).
* **Initialize with Pre-trained Weights:** Use `--weights <path/to/weights.pth>` to load pre-trained weights for transfer learning or fine-tuning.

```bash
# Example: Fine-tune previous best weights with a smaller learning rate
./venv/bin/python3 main.py --bootstrap --use-existing-dataset --weights task_20260630_144348/weights/best_segmenter.pth --lr 0.001 --epochs 15
```

### 2. Prediction, DFM Audit, and 3D Visual HUD
Segments any custom, user-provided `.stp` file. Loads the pre-trained GNN weights, classifies every face, evaluates engineering manufacturing rules, and launches the interactable Dual-Pane PyVista viewport.
```bash
./venv/bin/python3 main.py --predict data/synthetic/variant_1.stp
```
*Note: To specify custom weights, append `--weights /path/to/model.pth`.*

### 3. Procedural Dataset Generation Only
Generates synthetic CAD variants and matching JSON face-label manifests without running model training:
```bash
./venv/bin/python3 main.py --generate --num-variants 10
```

---

### 🧪 Running Verification Tests
Execute our pytest-driven verification suite to check compliance, data extraction pipelines, and neural dimension mappings:
```bash
./venv/bin/pytest -v
```

### 🧹 Formatting & Style Standards
Ruff is used to maintain style guidelines:
```bash
./venv/bin/ruff check . --fix
./venv/bin/ruff format .
```
