import numpy as np
import pyvista as pv
from pyvista import examples
import matplotlib.pyplot as plt

bunny = examples.download_bunny()

# Load robot arm

plotter = pv.Plotter(off_screen=True)

# FInd the coordinates of the center of the mesh
center = bunny.center
# Move the bunny to the origin
bunny.points -= center

# Change the camera position to view the bunny from the front
plotter.camera_position = 'xy'
plotter.camera.azimuth = 30
plotter.camera.elevation = 30
plotter.camera.zoom(2.5)

# add the bunny to the plotter
plotter.add_mesh(bunny)

plotter.show()

zval_filled_by_42s = plotter.get_image_depth()

# Plot the depth map
# Make background black
plt.imshow(zval_filled_by_42s)
plt.show()