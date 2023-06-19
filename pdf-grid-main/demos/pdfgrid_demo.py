from numpy import sqrt, array
from time import perf_counter
import matplotlib.pyplot as plt
from pdfgrid import PdfGrid
from pdfgrid.plotting import plot_marginal_2d


class ToroidalGaussian:
    def __init__(self):
        self.R0 = 1.
        self.eps = 0.05
        self.w2 = (self.R0*self.eps)**2

    def __call__(self, theta):
        x, y, z = theta
        r = sqrt(z**2 + (sqrt(x**2 + y**2) - self.R0)**2)
        return -0.5*r**2 / self.w2


posterior = ToroidalGaussian()
grid_spacing = array([0.04, 0.04, 0.02])
grid_centre = array([0., 0., 0.])
grid = PdfGrid(spacing=grid_spacing, offset=grid_centre)

# Main GridFill loop
t1 = perf_counter()
while grid.state != "end":
    P = array([posterior(theta) for theta in grid.get_parameters()])
    grid.give_probabilities(P)
t2 = perf_counter()

print(f"\n # RUNTIME: {(t2-t1)*1000:.1f} ms")

points, probs = grid.get_marginal(0)
plt.plot(points, probs)
plt.grid()
plt.tight_layout()
plt.show()

points, probs = grid.get_marginal([0, 1])
plot_marginal_2d(points, probs)

grid.plot_convergence()

from inference.plotting import matrix_plot

sample = grid.generate_sample(10000)
matrix_plot(sample.T)