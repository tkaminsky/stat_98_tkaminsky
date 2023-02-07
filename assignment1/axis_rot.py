import numpy as np
import matplotlib.pyplot as plt

# Import a bunny mesh
import pyvista as pv
from pyvista import examples

bunny = examples.download_bunny()

plotter = pv.Plotter()

# Find the coordinates of the center of the mesh

l = 1
start = (0, 0, 0)

# plotter.add_mesh(pv.Arrow(start, (l, 0, 0), tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='r', opacity=.2)
# plotter.add_mesh(pv.Arrow(start, (0, l, 0), tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='g', opacity=.2)
# plotter.add_mesh(pv.Arrow(start, (0, 0, l), tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='b', opacity=.2)
# Add a blue arrow from the origin to the point (0,1,0)


# Plot the bunny with the same scale as the arrows
bunny = bunny.scale(5)

# FInd the coordinates of the center of the mesh
center = bunny.center
# Move the bunny to the origin
bunny.points -= center

# Change the camera position to view the bunny from the front
plotter.camera_position = 'xy'

# Get the camera position
cam_pos = plotter.camera.position
print(cam_pos)
print(type(cam_pos))
# Move the camera back a bit
plotter.camera.position = (cam_pos[0], cam_pos[1], cam_pos[2] + 3)

plotter.camera.azimuth = 30
plotter.camera.elevation = 30


# Add a sphere with center at start and radius .1
# plotter.add_mesh(pv.Sphere(radius=1, center=start), color='r', opacity=.1)

# Make the background white
plotter.set_background('white')

# The rotation matrix for a rotation of 20 degrees around the y axis
R = np.array([[np.cos(np.pi/9), 0, np.sin(np.pi/9)],
              [0, 1, 0],
              [-np.sin(np.pi/9), 0, np.cos(np.pi/9)]])

initial_axis = [0,0,1]
# Rotate the initial axis by the rotation matrix
rotated_axis = R.dot(initial_axis)

# Rotate the [1,0,0], [0,1,0], and [0,0,1] to get the new axes
rotated_x = R.dot([1,0,0])
rotated_y = R.dot([0,1,0])
rotated_z = R.dot([0,0,1])

# Add the rotated axes to the plotter
plotter.add_mesh(pv.Arrow(start, rotated_x, tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='r')
plotter.add_mesh(pv.Arrow(start, rotated_y, tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='g')
plotter.add_mesh(pv.Arrow(start, rotated_z, tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='b')


# Draw an arc of a circle from [0,0,1] to rotated_axis
# arc = pv.CircularArc((0,0,1), rotated_axis, start, resolution=100)

# Add the arc to the plotter
# plotter.add_mesh(arc, color='blue', line_width=5)

# Add a golden arrow from the origin to the rotated axis
# plotter.add_mesh(pv.Arrow(start, rotated_axis, tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='gold')

# Rotate the bunny by 20 degrees around the y axis
bunny.rotate_y(20, inplace=True)

# Add the bunny to the plotter
plotter.add_mesh(bunny)

plotter.show()

# take a screenshot
plotter.screenshot('predicted_bunny.png')