import numpy as np
import matplotlib.pyplot as plt

np.random.seed(98)
# Generate 1000 samples from a normal distribution with mean 0 and standard deviation 1
samples = np.random.normal(0, 1, 1000)

b = 20

# Find the count of the samples in each bin
counts, bins = np.histogram(samples, bins=b)
# Find the largest count
max_count = np.max(counts)


# Plot the histogram of the samples
plt.hist(samples, bins=b, color='maroon')

mean_cands = [0.0, 0, .5]
std_cands = [1, .5, 1.5]

# Combine the mean and standard deviation candidates into a list of tuples

# Plot the normal pdf for each candidate mean and standard deviation over the histogram
for i in range(3):
    mean = mean_cands[i]
    std = std_cands[i]
    x = np.linspace(-5, 5, 1000)
    y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean)**2 / (2 * std**2))
    # Scale the pdf to match the largest count in the histogram
    y = y * max_count / np.max(y)
    plt.plot(x, y, linewidth=2)
        
# Clean up the plot
plt.xlabel('x')
plt.ylabel('Count')

# Remove the top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Remove all axes
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# Make the bottom spine thicker
ax.spines['bottom'].set_linewidth(2)



# Remove the ticks
ax.tick_params(bottom=False, left=False)

        
        
plt.show()