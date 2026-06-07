
import pyvista as pv

grid = pv.read("solution.vtu")

field = "u_abs"  

warped = grid.warp_by_scalar(field, factor=0.08)

plotter = pv.Plotter()

plotter.add_mesh(
    warped,
    scalars=field,
    show_edges=True,
    cmap="viridis"
)

plotter.add_title("Surface 3D "+ field +" avec maillage")
plotter.view_isometric()
plotter.show()