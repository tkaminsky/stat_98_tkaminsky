import numpy as np
import matplotlib.pyplot as plt

# Import a bunny mesh
import pyvista as pv
from pyvista import examples

bunny = examples.download_bunny()

plotter = pv.Plotter(off_screen=True)

# Find the coordinates of the center of the mesh

l = 1
start = (0, 0, 0)

plotter.add_mesh(pv.Arrow(start, (l, 0, 0), tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='r', opacity=.2)
plotter.add_mesh(pv.Arrow(start, (0, l, 0), tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='g', opacity=.2)
plotter.add_mesh(pv.Arrow(start, (0, 0, l), tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='b', opacity=.2)
# Add a blue arrow from the origin to the point (0,1,0)


# Plot the bunny with the same scale as the arrows
bunny = bunny.scale(5)

# FInd the coordinates of the center of the mesh
center = bunny.center
# Move the bunny to the origin
bunny.points -= center

# Change the camera position to view the bunny from the front
plotter.camera_position = 'xy'
plotter.camera.azimuth = 30
plotter.camera.elevation = 30

# Add a sphere with center at start and radius .1
plotter.add_mesh(pv.Sphere(radius=1, center=start), color='r', opacity=.1)

# Make the background white
plotter.set_background('white')

# The rotation matrix for a rotation of 20 degrees around the y axis
R = np.array([[np.cos(np.pi/9), 0, np.sin(np.pi/9)],
              [0, 1, 0],
              [-np.sin(np.pi/9), 0, np.cos(np.pi/9)]])

initial_axis = [0,0,1]
# Rotate the initial axis by the rotation matrix
rotated_axis = R.dot(initial_axis)



# Add a golden arrow from the origin to the rotated axis
plotter.add_mesh(pv.Arrow(start, rotated_axis, tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='gold')

import scipy as sc
import scipy.stats

import scipy.linalg as la

def sample_tangent_unit(mu):
    mat = np.matrix(mu)

    if mat.shape[1]>mat.shape[0]:
        mat = mat.T

    U,_,_ = la.svd(mat)
    nu = np.matrix(np.random.randn(mat.shape[0])).T
    x = np.dot(U[:,1:],nu[1:,:])
    return x/la.norm(x)

def rW(n, kappa, m):
    dim = m-1
    b = dim / (np.sqrt(4*kappa*kappa + dim*dim) + 2*kappa)
    x = (1-b) / (1+b)
    c = kappa*x + dim*np.log(1-x*x)

    y = []
    for i in range(0,n):
        done = False
        while not done:
            z = sc.stats.beta.rvs(dim/2,dim/2)
            w = (1 - (1+b)*z) / (1 - (1-b)*z)
            u = sc.stats.uniform.rvs()
            if kappa*w + dim*np.log(1-x*w) - c >= np.log(u):
                done = True
        y.append(w)
        
    return y

def rvMF(n,theta):
    dim = len(theta)
    kappa = np.linalg.norm(theta)
    mu = theta / kappa
    mu = mu.reshape((3,1))

    result = []
    w = rW(n, kappa, dim)
    for sample in range(0,n):
        v = sample_tangent_unit(mu).reshape((3,1))

        result.append(np.sqrt(1-w[sample]**2)*v + w[sample]*mu)

    return result

# Sample n points from a von Mises-Fisher distribution
n = 5000
kappa = 50
direction = rotated_axis
direction = direction / np.linalg.norm(direction)
points = rvMF(n, kappa * direction)
points = np.array(points).reshape((n, 3))
    

# Copy the bunny mesh
plotter.add_mesh(bunny)
bunny.rotate_y(20, inplace=True)
plotter.add_mesh(bunny)

# Plot each of the points as a small sphere
actor = None
actor_old = None

screenshots = [0,1,2,3,4,5,6,50,100,1000,2000,3000,4000,4999]

for i in range(0, n):
    # Draw a blue arrow to the point
    
    if i == 0:
        actor = plotter.add_mesh(pv.Arrow(start, points[i], tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='b')
        # Add a blue sphere at the point
    if i > 0:
        actor_old = actor 
        actor = plotter.add_mesh(pv.Arrow(start, points[i], tip_length=0.05, shaft_radius=.01, tip_radius=.02), color='b')
        
        # Remove the previous point
        plotter.remove_actor(actor_old)
        # Replace the previous point with a sphere
        plotter.add_mesh(pv.Sphere(radius=.01, center=points[i-1]), color='b', opacity=.5)

    # Take a screenshot and save it to the current directory
    if i in screenshots:
        plotter.screenshot('screenshot' + str(i) + '.png')
    