import vtk
import pyvista as pv
from typing import Callable, Tuple, Optional


class CADInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Custom VTK interactor style that maps Alt+LeftDrag to Rotate, Shift+LeftDrag to Pan, and LeftClick to Select."""

    def setup_style(
        self,
        plotter: pv.Plotter,
        mesh: pv.PolyData,
        on_select: Callable[[int, bool], None],
    ) -> None:
        """Initializes python properties and configures cell picker."""
        self.plotter = plotter
        self.mesh = mesh
        self.on_select = on_select
        self.click_pos: Optional[Tuple[int, int]] = None

        # Instantiate cell picker with standard VTK tolerance
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.025)

    def OnLeftButtonDown(self) -> None:
        self._on_left_down()

    def OnLeftButtonUp(self) -> None:
        self._on_left_up()

    def OnMouseMove(self) -> None:
        """Blocks standard camera rotation/pan when not holding Alt or Shift."""
        iren = self.GetInteractor()
        alt_pressed = iren.GetAltKey()
        shift_pressed = iren.GetShiftKey()
        if self.click_pos is not None and not alt_pressed and not shift_pressed:
            return
        super().OnMouseMove()

    def _on_left_down(self) -> None:
        iren = self.GetInteractor()
        self.click_pos = iren.GetEventPosition()
        alt_pressed = iren.GetAltKey()
        shift_pressed = iren.GetShiftKey()

        if alt_pressed or shift_pressed:
            # Let base class handle camera rotation/pan
            super().OnLeftButtonDown()
        else:
            # Selection mode: grab focus so OnLeftButtonUp is guaranteed
            self.GrabFocus(self.EventCallbackCommand)

    def _on_left_up(self) -> None:
        iren = self.GetInteractor()

        if self.click_pos is not None:
            release_pos = iren.GetEventPosition()

            renderer = iren.FindPokedRenderer(release_pos[0], release_pos[1])
            if renderer is None:
                renderer = self._find_fallback_renderer(release_pos)
            if renderer is not None:
                self.picker.Pick(release_pos[0], release_pos[1], 0, renderer)
                cell_id = self.picker.GetCellId()
            else:
                cell_id = -1

            if cell_id != -1:
                face_idx = int(self.mesh.cell_data["face_id"][cell_id])
                ctrl_pressed = bool(iren.GetControlKey())
                self.on_select(face_idx, ctrl_pressed)

            alt_pressed = iren.GetAltKey()
            shift_pressed = iren.GetShiftKey()

            if alt_pressed or shift_pressed:
                super().OnLeftButtonUp()
            else:
                self.ReleaseFocus()

            self.click_pos = None
        else:
            super().OnLeftButtonUp()

    def _find_fallback_renderer(self, pos: Tuple[int, int]) -> Optional[object]:
        """When FindPokedRenderer returns None, search all renderers."""
        iren = self.GetInteractor()
        rw = iren.GetRenderWindow()
        if rw is None:
            return None
        coll = rw.GetRenderers()
        coll.InitTraversal()
        r = coll.GetNextItem()
        while r is not None:
            vp = r.GetViewport()
            size = rw.GetSize()
            fx = pos[0] / size[0]
            fy = pos[1] / size[1]
            if vp[0] <= fx <= vp[2] and vp[1] <= fy <= vp[3]:
                return r
            r = coll.GetNextItem()
        return None
