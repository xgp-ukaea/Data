import matplotlib.pyplot as plt
from numpy import array, linspace, zeros, exp, log, pi, sqrt
from pdfgrid import PdfGrid
import time

# get the start time
st = time.time()

class Posterior:
    def __init__(self, x, y, y_err):
        # defining the parameters inside the class
        self.x = x
        self.y = y
        self.y_err = y_err
        self.t = 0.5

    # defining what is returned when the class is called
    def __call__(self, theta):
        return self.likelihood(theta) + self.prior(theta)

    def likelihood(self, theta):
        z = (self.y - self.model(self.x, theta)) / self.y_err
        logl = - log(self.y_err * (sqrt(2 * pi))).sum() - 0.5 * (z ** 2).sum()
        return logl

    def prior(self, theta):
        gradient, offset = theta
        return exp(-gradient / self.t)

    @staticmethod
    def model(x, theta):
        m, c = theta
        return m*x + c



# dataset we're analysing
x = array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
y = array([1.457, 2.466, 4.822, 4.33, 5.96, 6.222, 5.081, 7.575, 6.812, 7.426])
y_err = array([0.7, 0.907, 1.066, 1.2, 1.318, 1.425, 1.523, 1.614, 1.7, 1.781])

# feeding the input parameters into the class
posterior = Posterior(
    x=x,
    y=y,
    y_err=y_err
)

# grid_spacing and grid_centre set axis for grid, so they must be chosen to
# match the parameter for each axis
# for example, gradient is on x-axis, and we are interested of gradients between
# 0 and 1.5, so 0.75 is chosen as the first number in grid_centre
grid_spacing = array([0.025, 0.1])
grid_centre = array([0.75, 1.5])
grid = PdfGrid(spacing=grid_spacing, offset=grid_centre)

# Main GridFill loop, works as described in the lab book
while grid.state != "end":
    P = array([posterior(theta) for theta in grid.get_parameters()])
    grid.give_probabilities(P)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
milli = elapsed_time * 1000
print('\n Evaluation time:', milli, 'milliseconds')

indices, probs = grid.get_marginal(0)
plt.plot(indices*grid_spacing[0] + grid_centre[0], probs)
plt.grid()
plt.tight_layout()
plt.show()

params = [0, 1]
ind_axes, probs = grid.get_marginal(params)

ax1, ax2 = [ax*grid_spacing[p] for ax, p in zip(ind_axes, params)]
plt.contourf(ax1 + grid_centre[0], ax2 + grid_centre[1], probs.T)
plt.colorbar(label = 'Posterior')
plt.xlabel('Gradient')
plt.ylabel('Intercept')
plt.show()

grid.plot_convergence()

