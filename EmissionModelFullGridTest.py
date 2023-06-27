from numpy import array, log, linspace, unravel_index, meshgrid, stack, empty, ndindex
import matplotlib.pyplot as plt
from inference.plotting import matrix_plot
from agsi.priors import *
from agsi.likelihood import *
from numpy.random import choice
from scipy.integrate import simps
from pdfgrid import PdfGrid
from numpy.random import default_rng


def EvaluatedGrid(params, prob):
    # Creating a grid with dimensions i = parameter, j = parameter value
    meshgrid_params = meshgrid(*params, indexing='ij')
    # The stack function creates all combinations of n dimensions which are stored in comb_params
    comb_params = stack(meshgrid_params, axis=-1)

    # Evaluating the posterior at each point in the grid
    posterior_ND = prob(comb_params.reshape(-1, len(params)))

    posterior_ND = exp(posterior_ND)

    posterior_ND = posterior_ND.reshape((m, m, m, m, m, m))

    # Marginalisation routine
    marginals = []
    for f in range(len(params)):
        # Removes the parameter of interest from the dimensions being summed
        summed_dims = tuple(d for d in range(len(params)) if d != f)
        marginal = posterior_ND.sum(axis=summed_dims)
        marginal /= simps(marginal, x=params[f])
        marginals.append(marginal)

    return posterior_ND, marginals

"""
Generate synthetic data
"""
from agsi.models import BalmerTwoTemperature
model = BalmerTwoTemperature(lines=["D_alpha", "D_gamma", "D_delta"])
true_params = array([log(3.), log(0.8), log(1e19), log(0.5), 0.3, 0.12]).reshape([1, 6])
synthetic_emissions = model.prediction(true_params).squeeze()

EmissionData = SpecData(
    lines=["D_alpha", "D_gamma", "D_delta"],
    brightness=synthetic_emissions,
    uncertainty=synthetic_emissions*0.07
)


"""
Build posterior
"""
likelihood = SpecAnalysisLikelihood(
    measurements=EmissionData
)

HardLimit = BoundaryPrior(
    upper_limits=array([log(5), log(1), log(4e19), log(1), 0.98, 0.4]),
    lower_limits=array([log(0.2), log(0.2), log(5e18), log(1e-3), 0.01, 0.02])
)

ColdHotPrior = ColdTempPrior()

length_prior = PathLengthPrior(mode=0.1, lower_limit=0.05, upper_limit=0.2)

posterior = Posterior(
    components=[likelihood, HardLimit, ColdHotPrior, length_prior]
)

m = 10
hot_temp = linspace(log(0.2), log(5), m)
cold_temp = linspace(log(0.2), log(1), m)
e_density = linspace(log(5e18), log(4e19), m)
n_frac = linspace(log(1e-3), log(1), m)
mol_frac = linspace(0.01, 0.98, m)
path = linspace(0.02, 0.4, m)
params = (hot_temp, cold_temp, e_density, n_frac, mol_frac, path)

evaluated_prob, marginals = EvaluatedGrid((hot_temp, cold_temp, e_density, n_frac, mol_frac, path), posterior)

# Plotting marginals as subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()  # Flatten the subplots array for easier indexing
cols = ["red", "blue", "green", "purple", "orange", "black"]
for i, marginal in enumerate(marginals):
    axs[i].plot(params[i], marginal, color=cols[i])
    axs[i].set_xlabel('Parameter Value')
    axs[i].set_ylabel('Marginal Probability')
    axs[i].grid()

plt.tight_layout()
plt.show()

# Converts the posterior to 1-dimension and normalises
p = evaluated_prob.flatten()
p /= p.sum()

# Taking samples from the full grid based on probability weighting
index = choice(len(p), size=10000, p=p)

# Breaks the sample number down into the combination of the 6 parameters creating that cell
sample = unravel_index(index, (m, m, m, m, m, m))

matrix_plot(sample)
