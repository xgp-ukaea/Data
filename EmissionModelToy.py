import matplotlib.pyplot as plt
from numpy import array, linspace, zeros, exp, log, pi, sqrt
from pdfgrid import PdfGrid
from midas.emission import construct_emission_model
from inference.plotting import matrix_plot


class Posterior:
    def __init__(self, alpha, alpha_err, gamma, gamma_err, delta, delta_err):
        # defining the parameters inside the class
        self.alpha = alpha
        self.alpha_err = alpha_err
        self.gamma = gamma
        self.gamma_err = gamma_err
        self.delta = delta
        self.delta_err = delta_err

        self.alpha_model = construct_emission_model("D_alpha", is_include_mol_effects=True)
        self.gamma_model = construct_emission_model("D_gamma", is_include_mol_effects=True)
        self.delta_model = construct_emission_model("D_delta", is_include_mol_effects=True)

    # defining what is returned when the class is called
    def __call__(self, theta):
        return self.likelihood(theta)

    def likelihood(self, theta):
        x = (self.alpha - self.a_model(theta)) / self.alpha_err
        alpha_logl = - log(self.alpha_err * (sqrt(2 * pi))).sum() - 0.5 * (x ** 2).sum()
        y = (self.gamma - self.g_model(theta)) / self.gamma_err
        gamma_logl = - log(self.gamma_err * (sqrt(2 * pi))).sum() - 0.5 * (y ** 2).sum()
        z = (self.delta - self.d_model(theta)) / self.delta_err
        delta_logl = - log(self.delta_err * (sqrt(2 * pi))).sum() - 0.5 * (z ** 2).sum()
        return alpha_logl + gamma_logl + delta_logl

    def a_model(self, theta):
        path_length, n_e, n_frac, T_e_hot, T_e_cold, Q_mol = theta
        n_0 = n_frac * n_e
        return path_length * ((self.alpha_model.excitation(T_e_hot, n_e, n_0) + self.alpha_model.recombination(T_e_cold, n_e)) * (1 + Q_mol))

    def g_model(self, theta):
        path_length, n_e, n_frac, T_e_hot, T_e_cold, Q_mol = theta
        n_0 = n_frac * n_e
        return path_length * (self.gamma_model.excitation(T_e_hot, n_e, n_0) + self.gamma_model.recombination(T_e_cold, n_e) + self.gamma_model.molecular(T_e_hot, n_e, n_0, Q_mol))

    def d_model(self, theta):
        path_length, n_e, n_frac, T_e_hot, T_e_cold, Q_mol = theta
        n_0 = n_frac * n_e
        return path_length * (self.delta_model.excitation(T_e_hot, n_e, n_0) + self.delta_model.recombination(T_e_cold, n_e) + self.delta_model.molecular(T_e_hot, n_e, n_0, Q_mol))


intensity = Posterior(0, 0, 0, 0, 0, 0)  # Create an instance of AlphaPosterior
alpha_intensity = intensity.a_model((0.5, 1e19, 0.5, 4, 1, 1))  # Call the model method on the instance
gamma_intensity = intensity.g_model((0.5, 1e19, 0.5, 4, 1, 1))
delta_intensity = intensity.d_model((0.5, 1e19, 0.5, 4, 1, 1))
print(alpha_intensity, gamma_intensity, delta_intensity)

alpha = array([alpha_intensity*0.8, alpha_intensity*0.9, alpha_intensity, alpha_intensity*1.1, alpha_intensity*1.2])
alpha_err = array([alpha_intensity*0.05, alpha_intensity*0.05, alpha_intensity*0.05, alpha_intensity*0.05, alpha_intensity*0.05])
gamma = array([gamma_intensity*0.8, gamma_intensity*0.9, gamma_intensity, gamma_intensity*1.1, gamma_intensity*1.2])
gamma_err = array([gamma_intensity*0.05, gamma_intensity*0.05, gamma_intensity*0.05, gamma_intensity*0.05, gamma_intensity*0.05])
delta = array([delta_intensity*0.8, delta_intensity*0.9, delta_intensity, delta_intensity*1.1, delta_intensity*1.2])
delta_err = array([delta_intensity*0.05, delta_intensity*0.05, delta_intensity*0.05, delta_intensity*0.05, delta_intensity*0.05])

posterior = Posterior(
    alpha=alpha,
    alpha_err=alpha_err,
    gamma=gamma,
    gamma_err=gamma_err,
    delta=delta,
    delta_err=delta_err
)

grid_spacing = array([0.05, 1e18, 0.05, 0.5, 0.1, 0.1])
grid_centre = array([0.25, 2e17, 0.25, 2., 0.5, 0.5])
grid = PdfGrid(spacing=grid_spacing, offset=grid_centre)

while grid.state != "end":
    P = array([posterior(theta) for theta in grid.get_parameters()])
    grid.give_probabilities(P)

sample = grid.generate_sample(10000)
matrix_plot(sample.T)