from numpy import array, log, linspace, unravel_index, meshgrid, stack, empty, ndindex
import matplotlib.pyplot as plt
from inference.plotting import matrix_plot
from agsi.priors import *
from agsi.likelihood import *
from numpy.random import choice
from scipy.integrate import simps
from pdfgrid import PdfGrid
from numpy.random import default_rng
from numpy import shape
from time import perf_counter
import scipy.stats as scis
import matplotlib.mlab as mlab
import numpy as np


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

m = 12
hot_temp = linspace(log(0.2), log(5), m)
cold_temp = linspace(log(0.2), log(1), m)
e_density = linspace(log(5e18), log(4e19), m)
n_frac = linspace(log(1e-3), log(1), m)
mol_frac = linspace(0.01, 0.98, m)
path = linspace(0.02, 0.4, m)
params = (hot_temp, cold_temp, e_density, n_frac, mol_frac, path)

evaluated_prob, marginals = EvaluatedGrid((hot_temp, cold_temp, e_density, n_frac, mol_frac, path), posterior)

# Converts the posterior to 1-dimension and normalises
p = evaluated_prob.flatten()
p = array(p)
p /= p.sum()

# Taking samples from the full grid based on probability weighting
axis = [hot_temp, cold_temp, e_density, n_frac, mol_frac, path]
sampled_index = choice(len(p), size=50000, p=p)
meshgrid_index = meshgrid(*axis, indexing='ij')
comb_index = stack(meshgrid_index, axis=-1)
comb_index = comb_index.reshape(-1, len(params))
params = comb_index[sampled_index, :]

spacing = array([v[1] - v[0] for v in axis])

# Randomly pick points within the sampled cells
rng = default_rng()
full_sample = []
iter = 50000
full_sample = params + rng.uniform(
    low=-0.5*spacing,
    high=0.5*spacing,
    size=[50000, 6])
#resample if hard limit exceeded
selection_resample = np.all(np.logical_and(full_sample>HardLimit.lwr,full_sample<HardLimit.upr),axis=1)
while np.sum(selection_resample)>0:
    full_sample[selection_resample,:] = params[selection_resample,:] + rng.uniform(
        low=-0.5*spacing,
        high=0.5*spacing,
        size=[np.sum(selection_resample), 6])
    selection_resample = np.any(np.logical_not(np.logical_and(full_sample > HardLimit.lwr, full_sample < HardLimit.upr)), axis=1)


full_sample = array(full_sample)


def Kernal(model, sample, number, band, error=0.07):
    sample_emission = model.prediction(sample).squeeze()
    k = scis.gaussian_kde(sample_emission[number, :])
    xmin, xmax = min(sample_emission[number]), max(sample_emission[number])
    x = linspace(xmin, xmax, 10000)
    pdf_input = exp(- 0.5*((band - x)/(error*band))**2)
    pdf_input = pdf_input / sum(pdf_input)
    kde_eval = k(x)
    kde_eval = kde_eval / sum(kde_eval)

    return x, pdf_input, kde_eval


alpha_gaussian, pdf_input_Da, full_kde_eval_a = Kernal(model, full_sample, 0, synthetic_emissions[0])
gamma_gaussian, pdf_input_Dg, full_kde_eval_g = Kernal(model, full_sample, 1, synthetic_emissions[1])
delta_gaussian, pdf_input_Dd, full_kde_eval_d = Kernal(model, full_sample, 2, synthetic_emissions[2])

plt.plot(alpha_gaussian, full_kde_eval_a, 'b', label='Full Gridding Output')
plt.plot(alpha_gaussian, pdf_input_Da, 'r', label='Input')
plt.plot(gamma_gaussian, full_kde_eval_g, 'g', label='Gamma Output')
plt.plot(gamma_gaussian, pdf_input_Dg, 'y', label='Gamma Input')
plt.plot(delta_gaussian, full_kde_eval_d, color='black', label='Delta Output')
plt.plot(delta_gaussian, pdf_input_Dd, color='cyan', label='Delta Input')
plt.legend()
plt.show()
