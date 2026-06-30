import os
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, Dict, List

# OpenCascade Python bindings
import OCP.TopTools
import OCP.gp
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps


class CADGraphModel:
    """Extracts topological Graph Neural Network tensors from analytical B-Rep STEP files."""

    def __init__(self, step_path: str):
        if not os.path.exists(step_path):
            raise FileNotFoundError(f"STEP file not found: {step_path}")
        self.step_path = step_path
        self.shape, self.reader = self._load_step_shape(step_path)

    def _load_step_shape(
        self, step_path: str
    ) -> Tuple[TopoDS_Shape, STEPControl_Reader]:
        """Loads a STEP file and returns the root topological shape."""
        reader = STEPControl_Reader()
        if reader.ReadFile(step_path) != IFSelect_RetDone:
            raise IOError(f"Unable to read STEP entity: {step_path}")
        reader.TransferRoots()
        return reader.OneShape(), reader

    def extract_graph_tensors(self) -> Data:
        """Parses the loaded B-Rep shape boundaries into PyG Graph Tensors."""
        node_features, face_id_map, centroids = self._extract_face_features()
        edge_list, edge_features = self._extract_edges(face_id_map, node_features)

        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = (
            torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            if edge_list
            else torch.empty((2, 0), dtype=torch.long)
        )
        edge_attr = (
            torch.tensor(edge_features, dtype=torch.float)
            if edge_features
            else torch.empty((0, 1), dtype=torch.float)
        )

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.face_id_map = face_id_map
        data.centroids = centroids
        return data

    def _extract_face_features(
        self,
    ) -> Tuple[List[List[float]], Dict[int, int], List[List[float]]]:
        """Extracts node features from each analytical B-Rep face."""
        face_explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        node_features: List[List[float]] = []
        centroids: List[List[float]] = []
        face_id_map: Dict[int, int] = {}
        idx = 0

        while face_explorer.More():
            face = TopoDS.Face(face_explorer.Current())
            f_hash = hash(face)
            face_id_map[f_hash] = idx

            # Compute physical area and centroid of the face
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            area = props.Mass()
            centroid = props.CentreOfMass()
            centroids.append([centroid.X(), centroid.Y(), centroid.Z()])

            # Extract underlying geometry surface type
            surf_adaptor = BRepAdaptor_Surface(face)
            surf_type = surf_adaptor.GetType().value

            # Extract surface normal at parametric center
            normal = self._get_face_normal(surf_adaptor)

            # Feature vector: [surface_type, area, normal_x, normal_y, normal_z, centroid_z]
            feat = [
                float(surf_type),
                float(area),
                normal.X(),
                normal.Y(),
                normal.Z(),
                centroid.Z(),
            ]
            node_features.append(feat)

            idx += 1
            face_explorer.Next()

        return node_features, face_id_map, centroids

    def _get_face_normal(self, surf_adaptor: BRepAdaptor_Surface) -> OCP.gp.gp_Dir:
        """Evaluates and returns the surface normal vector at parametric center."""
        u_min, u_max = surf_adaptor.FirstUParameter(), surf_adaptor.LastUParameter()
        v_min, v_max = surf_adaptor.FirstVParameter(), surf_adaptor.LastVParameter()
        props_lp = BRepLProp_SLProps(
            surf_adaptor, (u_min + u_max) / 2.0, (v_min + v_max) / 2.0, 1, 1e-7
        )
        return (
            props_lp.Normal() if props_lp.IsNormalDefined() else OCP.gp.gp_Dir(0, 0, 1)
        )

    def _extract_edges(
        self, face_id_map: Dict[int, int], node_features: List[List[float]]
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """Traverses the edge-face adjacency map to build edge links."""
        edge_to_faces_map = OCP.TopTools.TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(
            self.shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces_map
        )

        edge_list: List[List[int]] = []
        edge_features: List[List[float]] = []

        for e_idx in range(1, edge_to_faces_map.Extent() + 1):
            face_list = edge_to_faces_map.FindFromIndex(e_idx)
            if face_list.Extent() == 2:  # Shared boundary transition
                f1 = TopoDS.Face(face_list.First())
                f2 = TopoDS.Face(face_list.Last())

                idx1 = face_id_map.get(hash(f1))
                idx2 = face_id_map.get(hash(f2))

                if idx1 is not None and idx2 is not None:
                    self._add_edge_transition(
                        idx1, idx2, node_features, edge_list, edge_features
                    )

        return edge_list, edge_features

    def _add_edge_transition(
        self,
        idx1: int,
        idx2: int,
        node_features: List[List[float]],
        edge_list: List[List[int]],
        edge_features: List[List[float]],
    ) -> None:
        """Computes dihedral angle and appends bidirectional edges to lists."""
        # Standard edge connectivity
        edge_list.append([idx1, idx2])
        edge_list.append([idx2, idx1])

        # Feature metric: Dihedral angle between adjacent face normals
        n1 = np.array(node_features[idx1][2:5])
        n2 = np.array(node_features[idx2][2:5])
        dihedral = float(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))

        edge_features.append([dihedral])
        edge_features.append([dihedral])
