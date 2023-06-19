import matplotlib.pyplot as plt
import numpy as np

def gaussian(x, mean, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])
y_err = np.array([0.3, 0.3, 0.3, 0.3])

y_1 = (1.15 * x - 0.4)
y_2 = (0.85 * x + 0.4)

fig, ax = plt.subplots()

# Plot scatter points and error bars
ax.scatter(x, y)
ax.errorbar(x, y, yerr=y_err, ls="none")

# Plot Gaussians along each error bar
for i, (xi, yi, err) in enumerate(zip(x, y, y_err)):
    x_gaussian = np.linspace(xi - 3 * err, xi + 3 * err, 100)
    y_gaussian = gaussian(x_gaussian, yi, err)
    ax.plot(x_gaussian, y_gaussian)

# Plot the lines y_1 and y_2
ax.plot(x, y_1)
ax.plot(x, y_2)

plt.show()
v#