import numpy as np
import pyvista as pv
from typing import Tuple, Dict, List

# OpenCascade structural python bindings
import OCP.TopLoc
import OCP.TopAbs
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh


class OCPMesher:
    """Tessellates analytical OpenCascade TopoDS_Shape into PyVista meshes."""

    @staticmethod
    def tessellate_shape(
        shape: TopoDS_Shape, deflection: float = 0.1
    ) -> Tuple[pv.PolyData, Dict[int, int]]:
        """Tessellates B-Rep boundaries and maps triangles to topological face indices.

        Args:
            shape: Native TopoDS_Shape loaded from a CAD engine.
            deflection: Linear deflection tolerance for triangulation (default: 0.1mm).

        Returns:
            A tuple of (pyvista.PolyData mesh, face_hash_to_index_map).
        """
        # Build incremental triangular mesh using OpenCascade mesh generator
        BRepMesh_IncrementalMesh(shape, deflection).Perform()

        points_list: List[List[float]] = []
        faces_list: List[List[int]] = []
        cell_face_ids: List[int] = []
        face_hash_to_idx: Dict[int, int] = {}

        face_explorer = TopExp_Explorer(shape, OCP.TopAbs.TopAbs_FACE)
        face_idx = 0

        while face_explorer.More():
            face = TopoDS.Face(face_explorer.Current())
            f_hash = hash(face)
            face_hash_to_idx[f_hash] = face_idx

            loc = OCP.TopLoc.TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, loc)

            if triangulation is not None:
                OCPMesher._extract_triangulation(
                    triangulation, loc, points_list, faces_list, cell_face_ids, face_idx
                )

            face_idx += 1
            face_explorer.Next()

        # Handle empty geometry case safely
        if not points_list:
            return pv.PolyData(), face_hash_to_idx

        # Build visual PolyData mesh
        points_arr = np.array(points_list, dtype=np.float32)
        faces_arr = (
            np.concatenate(faces_list) if faces_list else np.array([], dtype=np.int32)
        )
        mesh = pv.PolyData(points_arr, faces_arr)
        mesh.cell_data["face_id"] = np.array(cell_face_ids, dtype=np.int32)

        return mesh, face_hash_to_idx

    @staticmethod
    def _extract_triangulation(
        triangulation,
        loc: OCP.TopLoc.TopLoc_Location,
        points_list: List[List[float]],
        faces_list: List[List[int]],
        cell_face_ids: List[int],
        face_idx: int,
    ) -> None:
        """Helper to extract points and triangles from a face triangulation.

        Keeps functions modular and under 30 lines.
        """
        nb_nodes = triangulation.NbNodes()
        nb_triangles = triangulation.NbTriangles()
        node_offset = len(points_list)

        transf = loc.Transformation()
        for i in range(1, nb_nodes + 1):
            p = triangulation.Node(i)
            p_trans = p.Transformed(transf)
            points_list.append([p_trans.X(), p_trans.Y(), p_trans.Z()])

        for i in range(1, nb_triangles + 1):
            tri = triangulation.Triangle(i)
            idx1, idx2, idx3 = tri.Value(1), tri.Value(2), tri.Value(3)
            # PyVista format: [n_points, p1, p2, p3]
            faces_list.append(
                [
                    3,
                    node_offset + idx1 - 1,
                    node_offset + idx2 - 1,
                    node_offset + idx3 - 1,
                ]
            )
            cell_face_ids.append(face_idx)
