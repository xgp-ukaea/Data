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

        self.emission_models = {
            "alpha": construct_emission_model("D_alpha", is_include_mol_effects=True),
            "gamma": construct_emission_model("D_gamma", is_include_mol_effects=True),
            "delta": construct_emission_model("D_delta", is_include_mol_effects=True)
        }

    def __call__(self, theta):
        return self.likelihood(theta) + self.hard_priors(theta)

    def likelihood(self, theta):
        for emission_type in self.emission_models:
            if self.hard_priors(theta) == 0:
                x = (self._get_emission_value(emission_type) - self.Balmer_model(emission_type, theta)) / self._get_emission_error(emission_type)
                log_likelihood = - log(self._get_emission_error(emission_type) * sqrt(2 * pi)).sum() - 0.5 * (x ** 2).sum()

                return log_likelihood

            else:
                return 0

    def hard_priors(self, theta):
        path_length, n_e, n_frac, T_e_hot, T_e_cold, f_mol = theta

        if path_length < 0.02 or path_length > 0.4:
            return -1e50
        elif T_e_hot < 0.2 or T_e_hot > 5:
            return -1e50
        elif T_e_cold < 0.02 or T_e_cold > 1:
            return -1e50
        elif T_e_hot <= T_e_cold:
            return -1e50
        elif f_mol < 0.01 or f_mol > 0.99:
            return -1e50
        elif n_e < 1e17 or n_e > 1e21:
            return -1e50
        elif n_frac < 1e-3 or n_frac > 100:
            return -1e50
        else:
            return 0

    def Balmer_model(self, emission_type, theta):
        path_length, n_e, n_frac, T_e_hot, T_e_cold, f_mol = theta

        if self.hard_priors(theta) == 0:
            emission_model = self.emission_models[emission_type]

            n_e = log(n_e)
            n_frac = log(n_frac)
            T_e_hot = log(T_e_hot)
            T_e_cold = log(T_e_cold)

            n_0 = n_frac * n_e
            Q_mol = f_mol / (1 - f_mol)

            return path_length * (
                    emission_model.excitation(T_e_hot, n_e, n_0) +
                    emission_model.recombination(T_e_cold, n_e) +
                    emission_model.molecular(T_e_hot, n_e, n_0, Q_mol)
            )

        else:
            return 0

    def _get_emission_value(self, emission_type):
        return getattr(self, emission_type)

    def _get_emission_error(self, emission_type):
        return getattr(self, f"{emission_type}_err")


alpha_intensity = 1e20
gamma_intensity = 1e19
delta_intensity = 1e18

alpha = array([alpha_intensity * 0.8, alpha_intensity * 0.9, alpha_intensity, alpha_intensity * 1.1, alpha_intensity * 1.2])
alpha_err = array([alpha_intensity * 0.05, alpha_intensity * 0.05, alpha_intensity * 0.05, alpha_intensity * 0.05, alpha_intensity * 0.05])
gamma = array([gamma_intensity * 0.8, gamma_intensity * 0.9, gamma_intensity, gamma_intensity * 1.1, gamma_intensity * 1.2])
gamma_err = array([gamma_intensity * 0.05, gamma_intensity * 0.05, gamma_intensity * 0.05, gamma_intensity * 0.05, gamma_intensity * 0.05])
delta = array([delta_intensity * 0.8, delta_intensity * 0.9, delta_intensity, delta_intensity * 1.1, delta_intensity * 1.2])
delta_err = array([delta_intensity * 0.05, delta_intensity * 0.05, delta_intensity * 0.05, delta_intensity * 0.05, delta_intensity * 0.05])

posterior = Posterior(
    alpha=alpha,
    alpha_err=alpha_err,
    gamma=gamma,
    gamma_err=gamma_err,
    delta=delta,
    delta_err=delta_err
)

grid_spacing = array([0.02, 1e18, 0.05, 0.5, 0.1, 0.1])
grid_centre = array([0.1, 5e18, 0.25, 1.25, 0.25, 0.25])
grid = PdfGrid(spacing=grid_spacing, offset=grid_centre)

while grid.state != "end":
    P = array([posterior(theta) for theta in grid.get_parameters()])
    grid.give_probabilities(P)

sample = grid.generate_sample(10000)
matrix_plot(sample.T)
