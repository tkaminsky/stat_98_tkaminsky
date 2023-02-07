import numpy as np
import matplotlib.pyplot as plt

# Visualize the von mises pdf over a unit circle

def vmf_pdf(x, mu, kappa):
    return np.exp(kappa * mu.dot(x)) / (2 * np.pi * np.sinh(kappa))

# Generate a grid of points on the unit circle
n = 100
theta = np.linspace(0, 2 * np.pi, n)
x = np.cos(theta)
y = np.sin(theta)
z = np.zeros(n)

# Evaluate the pdf at each point
kappa = 20
mu = np.array([- np.sqrt(2) / 2, np.sqrt(2) / 2])
pdf = np.array([vmf_pdf(np.array([x[i], y[i]]), mu, kappa) for i in range(n)])
pdf += .1

# Plot the pdf in a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, c='k')
ax.plot(x, y, pdf, c='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Remove numbers from the axes
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_zaxis().set_visible(False)

# Remove axes
ax.w_xaxis.line.set_visible(False)
ax.w_yaxis.line.set_visible(False)
ax.w_zaxis.line.set_visible(False)

# Remove background
ax.set_facecolor('none')

# Remove the grid
ax.grid(False)

# Remove ticks
ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

# Make the 3D background transparent
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove ticks from all 3D axes
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Remove tick marks
ax.xaxis._axinfo['tick']['inward_factor'] = 0
ax.xaxis._axinfo['tick']['outward_factor'] = 0
ax.yaxis._axinfo['tick']['inward_factor'] = 0
ax.yaxis._axinfo['tick']['outward_factor'] = 0
ax.zaxis._axinfo['tick']['inward_factor'] = 0
ax.zaxis._axinfo['tick']['outward_factor'] = 0



# Remove the spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)




# set the axes limits
ax.set_zlim(0, .7)
plt.show()
