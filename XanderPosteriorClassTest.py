from numpy import array, linspace, zeros, exp, log, pi, sqrt, ndindex, meshgrid, stack, empty
from scipy.integrate import simps
import matplotlib.pyplot as plt

class Posterior:
    def __init__(self, x, y, y_err):
        # defining the parameters that are used inside the class
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
        return log(exp(-gradient / self.t))

    @staticmethod
    def model(x, theta):
        m, c = theta
        return m*x + c


def EvaluatedGrid(params, prob):
    # Creating a grid with dimensions i = parameter, j = parameter value
    meshgrid_params = meshgrid(*params, indexing='ij')
    # The stack function creates all combinations of n dimensions which are stored in comb_params
    comb_params = stack(meshgrid_params, axis=-1)
    posterior_ND = empty(comb_params.shape[:-1])

    # Evaluating the posterior at each point in the grid
    for index in ndindex(*comb_params.shape[:-1]):
        posterior_ND[index] = prob(comb_params[index])

    posterior_ND = exp(posterior_ND)

    # Marginalisation routine
    for f in range(len(params)):
        # Removes the parameter of interest from the dimensions being summed
        summed_dims = tuple(d for d in range(len(params)) if d != f)
        marginal = posterior_ND.sum(axis=(summed_dims))
        marginal /= simps(marginal, x=params[f])
        plt.plot(params[f], marginal)

    return posterior_ND, marginal


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

m = 128
gradient_axis = linspace(0, 1.5, m)
offset_axis = linspace(-2, 5, m)

evaluated_prob, marginals = EvaluatedGrid((gradient_axis, offset_axis), posterior)

plt.grid()
plt.tight_layout()
plt.show()

plt.contourf(gradient_axis, offset_axis, evaluated_prob.T)
plt.colorbar(label = 'Posterior')
plt.show()
