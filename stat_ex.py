import numpy as np
import matplotlib.pyplot as plt

np.random.seed(98)

# Generate random data from an exponential distribution with a mean of 45
data = np.random.exponential(scale=45, size=365)
print(f"Our data has a mean of {np.mean(data)}")

# Add horizontal grid lines
plt.grid(which='major', axis='y', linestyle='-', color='gray', alpha=0.7, zorder=1)

# Plot the data as a histogram and clean up the plot
plt.hist(data, bins=20, color='orange', edgecolor='black', linewidth=1., zorder=2)

# Make the axes look nicer
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
plt.title('Bus Wait Times Over One Year', fontdict={'fontsize': 20, 'fontname': 'Times New Roman'})
plt.xlabel('Wait Time (minutes)', fontdict={'fontname': 'Times New Roman', 'fontsize': 15})
plt.ylabel('Frequency', fontdict={'fontname': 'Times New Roman', 'fontsize': 15})
plt.show()

# Part 2:
#   Plotting the likelihood of the data as an exponential distribution
#
#

# Returns the likelihood of the data given l (the mean)
def get_likelihood(l):
    # Store the data as a high-precision data type so likelihood isn't just 0
    data_precise = np.array(data, dtype=np.float128)
    # Calculate the likelihood with high precision
    likelihood = np.prod(np.exp(-data_precise / l) / l)
    return likelihood

# Find the likelihood for l between 0 and 100 as a high-precision data type
likelihoods = np.array([get_likelihood(l) for l in np.linspace(35, 60, 1000)], dtype=np.float128)

# Find the index of the maximum likelihood
max_index = np.argmax(likelihoods)
# Find the value of l that corresponds to the maximum likelihood
max_l = np.linspace(35, 60, 1000)[max_index]
print("Max likelihood is at l = {}".format(max_l))

# scale the likelihoods so they are between 0 and 1 (to make graphing feasible)
likelihoods = likelihoods / np.max(likelihoods)

# Plot the likelihoods
plt.plot(np.linspace(20, 80, 1000), likelihoods, color='orange', linewidth=2, zorder=2)

# Clean up the graphs
plt.title('Likelihood for Mean Wait Time ', fontdict={'fontsize': 20, 'fontname': 'Times New Roman'})
plt.xlabel('Mean Wait Time (minutes)', fontdict={'fontname': 'Times New Roman', 'fontsize': 15})
plt.ylabel('Likelihood of Our Data (really small)', fontdict={'fontname': 'Times New Roman', 'fontsize': 15})
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
plt.grid(which='major', axis='y', linestyle='-', color='gray', alpha=0.7, zorder=1)
plt.gca().set_yticklabels([])

plt.show()

# Part 3:
#   Graphing sample exponential distributions with different means
#

# Grab 5 exponential distributions with different means
for i, l in enumerate([30, 40, 50, 60, 70]):
    x = np.linspace(0, 200, 1000)
    y = np.exp(-x / l) / l
    # Plot the current line
    plt.plot(x, y, color='C{}'.format(i), linewidth=2, zorder=3)

# Clean up the plot
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
plt.title('Exponential Distributions with Different Means', fontdict={'fontsize': 20, 'fontname': 'Times New Roman'})
plt.xlabel('Wait Time', fontdict={'fontname': 'Times New Roman', 'fontsize': 15})
plt.ylabel('Probability Density', fontdict={'fontname': 'Times New Roman', 'fontsize': 15})
plt.grid(which='major', axis='y', linestyle='-', color='gray', alpha=0.7, zorder=1)
plt.legend(['30', '40', '50', '60', '70'], loc='upper right', frameon=False)

plt.gca().get_legend().set_title('Mean Wait Time', prop={'size': 10, 'family': 'Times New Roman'})

plt.show()


