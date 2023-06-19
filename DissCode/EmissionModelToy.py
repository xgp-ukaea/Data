import matplotlib.pyplot as plt
from numpy import array, linspace, zeros, exp, log, pi, sqrt
from pdfgrid import PdfGrid
from midas.emission import construct_emission_model
from inference.plotting import matrix_plot


class AlphaPosterior:
    def __init__(self, alpha, alpha_err):
        # defining the parameters inside the class
        self.alpha = alpha
        self.alpha_err = alpha_err

        self.alpha_model = construct_emission_model("D_alpha", is_include_mol_effects=True)

    # defining what is returned when the class is called
    def __call__(self, theta):
        return self.likelihood(theta)

    def likelihood(self, theta):
        z = (self.alpha - self.model(theta)) / self.alpha_err
        logl = - log(self.alpha_err * (sqrt(2 * pi))).sum() - 0.5 * (z ** 2).sum()
        return logl

    def model(self, theta):
        path_length, n_e, n_frac, T_e_hot, T_e_cold, Q_mol = theta
        n_0 = n_frac * n_e
        return path_length * ((self.alpha_model.excitation(T_e_hot, n_e, n_0) + self.alpha_model.recombination(T_e_cold, n_e)) * (1 + Q_mol))


alpha_posterior = AlphaPosterior(0, 0)  # Create an instance of AlphaPosterior
data = alpha_posterior.model((0.5, 1e19, 0.5, 5, 1, 1))  # Call the model method on the instance
print(data)
# The value for balmer line emissions for this case is 1.27e+22

y = array([1., 1.1, 1.2, 1.3, 1.4])
y *= 1e22
y_err = array([0.01, 0.01, 0.01, 0.01, 0.01])
y_err *= 1e22

posterior = AlphaPosterior(
    alpha=y,
    alpha_err=y_err
)

grid_spacing = array([0.05, 1e18, 0.05, 0.5, 0.1, 0.1])
grid_centre = array([0.25, 5e18, 0.25, 2.5, 0.5, 0.5])
grid = PdfGrid(spacing=grid_spacing, offset=grid_centre)

while grid.state != "end":
    P = array([posterior(theta) for theta in grid.get_parameters()])
    grid.give_probabilities(P)


sample = grid.generate_sample(10000)
matrix_plot(sample.T)