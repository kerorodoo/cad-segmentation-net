import os
import json
import random
import math
from typing import Tuple, Dict, List

# CadQuery procedural geometry toolbox
import cadquery as cq

# OpenCascade Python bindings
import OCP.TopAbs
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepAdaptor import BRepAdaptor_Surface


class StructuralDataFactory:
    """Generates synthetic training sets with perfect topological ground truth."""

    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build_variant(self, serial_id: int) -> Tuple[str, str]:
        """Generates a random procedural CAD chassis and returns paths (step, labels)."""
        base_w = random.uniform(80.0, 120.0)
        base_d = random.uniform(40.0, 60.0)
        base_h = random.uniform(1.5, 2.5)

        boss_h = random.uniform(6.0, 14.0)
        boss_dia = random.uniform(5.0, 8.0)

        # Build solid part containing all 7 classes
        part = self._build_geometry(base_w, base_d, base_h, boss_h, boss_dia)

        # Save STEP file
        step_path = os.path.join(self.output_dir, f"variant_{serial_id}.stp")
        cq.exporters.export(part, step_path, cq.exporters.ExportTypes.STEP)

        # Analyze shape and write labels manifest
        occt_shape = part.findSolid().wrapped
        labels_manifest = self._generate_labels(occt_shape, base_h, base_w, base_d)

        manifest_path = os.path.join(
            self.output_dir, f"variant_{serial_id}_labels.json"
        )
        with open(manifest_path, "w") as f:
            json.dump(labels_manifest, f, indent=2)

        return step_path, manifest_path

    def _build_geometry(
        self,
        base_w: float,
        base_d: float,
        base_h: float,
        boss_h: float,
        boss_dia: float,
    ) -> cq.Workplane:
        """Procedurally models the chassis base, screw bosses, text, symbols, and ribs."""
        # 1. Plate (Class 0)
        part = cq.Workplane("XY").box(base_w, base_d, base_h)

        # 2. Add Hole (Class 4) through the plate
        part = (
            part.faces(">Z").workplane().center(-base_w / 4.0, -base_d / 4.0).hole(4.0)
        )

        # 3. Add Embossed Text "CAD" (Class 5 - zifu_text) at positive X
        part = (
            part.faces(">Z").workplane().center(base_w / 6.0, 0.0).text("CAD", 8.0, 0.5)
        )

        # 4. Add Embossed Polygon Symbol (Class 6 - zifu_symbol) at negative X
        part = (
            part.faces(">Z")
            .workplane()
            .center(-base_w / 6.0, 0.0)
            .polygon(5, 8.0)
            .extrude(0.5)
        )

        # 5. Inject Screw Bosses (Class 2) on top face (>Z)
        top_wp = part.faces(">Z").workplane()
        top_wp = self._place_bosses(top_wp, base_w, base_d, boss_dia, boss_h)

        # 6. Add a Clip/Hook (Class 3)
        top_wp = self._place_clip(top_wp, base_w, base_d)

        # 7. Inject Structural Stiffener Ribs (Class 1) on bottom face (<Z)
        bottom_wp = (
            cq.Workplane("XY")
            .newObject([top_wp.findSolid()])
            .faces("<Z")
            .workplane()
            .center(0, 0)
        )
        return self._draw_ribs(bottom_wp, base_w, base_d)

    def _place_bosses(
        self,
        top_wp: cq.Workplane,
        base_w: float,
        base_d: float,
        boss_dia: float,
        boss_h: float,
    ) -> cq.Workplane:
        """Places up to 2 non-overlapping cylindrical bosses on the top face."""
        num_bosses = random.randint(1, 2)
        boss_positions: List[Tuple[float, float]] = []

        # Find safe coordinates on the top surface
        for _ in range(num_bosses):
            for _ in range(30):
                # Place columns on negative Y quadrant to avoid clip and text
                bx = random.uniform(
                    -base_w / 2.0 + boss_dia * 1.5, base_w / 2.0 - boss_dia * 1.5
                )
                by = random.uniform(-base_d / 2.0 + boss_dia * 1.5, -boss_dia * 1.5)

                if all(
                    math.sqrt((bx - px) ** 2 + (by - py) ** 2) >= (boss_dia + 3.0)
                    for px, py in boss_positions
                ):
                    boss_positions.append((bx, by))
                    break

        # Extrude each boss
        for bx, by in boss_positions:
            top_wp = (
                top_wp.center(bx, by)
                .circle(boss_dia / 2.0)
                .circle(boss_dia / 3.0)
                .extrude(boss_h)
                .center(-bx, -by)
            )

        return top_wp

    def _place_clip(
        self, top_wp: cq.Workplane, base_w: float, base_d: float
    ) -> cq.Workplane:
        """Models a planar vertical L-shaped overhang clip (Class 3)."""
        hook_x = base_w / 4.0
        hook_y = base_d / 4.0

        # Vertical post
        top_wp = (
            top_wp.center(hook_x, hook_y)
            .rect(6.0, 4.0)
            .extrude(8.0)
            .center(-hook_x, -hook_y)
        )

        # Overlay horizontal lip on top of the vertical post
        top_wp = (
            cq.Workplane("XY")
            .newObject([top_wp.findSolid()])
            .faces(">Z")
            .workplane()
            .center(hook_x, hook_y + 1.0)
            .rect(6.0, 2.0)
            .extrude(2.0)
            .center(-hook_x, -(hook_y + 1.0))
        )
        return top_wp

    def _draw_ribs(
        self, bottom_wp: cq.Workplane, base_w: float, base_d: float
    ) -> cq.Workplane:
        """Draws randomized rib structures (horizontal, vertical, parallel, cross)."""
        rib_style = random.choice(["horizontal", "vertical", "grid", "cross"])

        if rib_style == "horizontal":
            bottom_wp = bottom_wp.rect(base_w * random.uniform(0.7, 0.9), 2.0)
        elif rib_style == "vertical":
            bottom_wp = bottom_wp.rect(2.0, base_d * random.uniform(0.7, 0.9))
        elif rib_style == "grid":
            offset = random.uniform(8.0, 12.0)
            bottom_wp = (
                bottom_wp.center(0, -offset).rect(base_w * 0.8, 2.0).center(0, offset)
            )
            bottom_wp = (
                bottom_wp.center(0, offset).rect(base_w * 0.8, 2.0).center(0, -offset)
            )
        elif rib_style == "cross":
            bottom_wp = bottom_wp.rect(base_w * 0.8, 2.0).rect(2.0, base_d * 0.8)

        return bottom_wp.extrude(4.0)

    def _generate_labels(
        self, occt_shape, base_h: float, base_w: float, base_d: float
    ) -> Dict[str, int]:
        """Maps topological face indices to semantic integer categories based on com Z."""
        explorer = TopExp_Explorer(occt_shape, OCP.TopAbs.TopAbs_FACE)
        labels_manifest: Dict[str, int] = {}
        idx = 0

        while explorer.More():
            face = TopoDS.Face(explorer.Current())

            # Fetch evaluation metrics to resolve component class
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            com = props.CentreOfMass()

            surf_adaptor = BRepAdaptor_Surface(face)
            surf_type = surf_adaptor.GetType().value

            assigned_class = 0  # Default Chassis Plate

            if com.Z() < (-base_h / 2.0 - 0.1):
                # Stiffener Rib (Class 1)
                assigned_class = 1
            elif com.Z() > (base_h / 2.0 + 0.1):
                # Above the plate
                # Check for Zifu (text & symbols) first (close to top surface of base)
                if com.Z() <= base_h / 2.0 + 0.6:
                    assigned_class = 5  # zifu(letter, character, word)
                # Check for Clip bounds (Positive X & Y Quadrant)
                elif (com.X() > 0) and (com.Y() > 0):
                    assigned_class = 3  # Clip/Hook
                else:
                    assigned_class = 2  # Column/Boss
            else:
                # Inside plate height
                if surf_type == 1:
                    assigned_class = 4  # Hole (Cylindrical cut)
                else:
                    assigned_class = 0  # Plate

            labels_manifest[str(idx)] = assigned_class
            idx += 1
            explorer.Next()

        return labels_manifest


def generate_single_variant_process(args: Tuple[str, int]) -> Tuple[str, str]:
    """Pickleable multiprocessing worker to generate a procedural variant."""
    output_dir, serial_id = args
    factory = StructuralDataFactory(output_dir)
    return factory.build_variant(serial_id)
