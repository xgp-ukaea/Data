from numpy import array, log, meshgrid, empty, linspace, stack, load
from pdfgrid import PdfGrid
from inference.plotting import matrix_plot
from agsi.priors import *
from agsi.likelihood import *
from agsi.models import BalmerTwoTemperature
from numpy.random import default_rng, choice
import matplotlib.pyplot as plt
from scipy.integrate import simps
import scipy.stats as scis
import numpy as np
from time import perf_counter
import dms.analysis.emission.Balmer_analysis as BA

def ImportData(input):
    input_data = load(input, allow_pickle=True)
    input_data = input_data[()]
    D_alpha = input_data['n1Int']
    D_beta = input_data['n2Int']
    Uncertainty = input_data['AbsErr']
    length_mode = input_data['DL']
    length_lower = input_data['DLL']
    length_upper = input_data['DLH']
    fulcher = input_data['Fulcher']
    fulcherLimits, te_lim_low, te_lim_high = BA.get_fulcher_constraint_spline(fulcher, telim=True)
    fulcherLimits = fulcherLimits
    DenMean = input_data['Den']
    DenErr = input_data['DenErr']
    input_data = 0

    return D_alpha, D_beta, Uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits

D_alpha, D_beta, uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits = ImportData("input.npy")
emission_time = 0
LoS = 0
simulated_data = array([D_alpha[emission_time, LoS], D_beta[emission_time, LoS]])

EmissionData = SpecData(
    lines=["D_alpha", "D_beta"],
    brightness=simulated_data,
    uncertainty=simulated_data*uncertainty
)

from agsi.models import BalmerTwoTemperature
model = BalmerTwoTemperature(lines=["D_alpha", "D_beta"])


def EvaluatedGrid(params, prob):
    # Creating a grid with dimensions i = parameter, j = parameter value
    meshgrid_params = meshgrid(*params, indexing='ij')
    # The stack function creates all combinations of n dimensions which are stored in comb_params
    comb_params = stack(meshgrid_params, axis=-1)
    posterior_ND = empty(comb_params.shape[:-1])

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

length_prior = PathLengthPrior(mode=length_mode[emission_time, LoS], lower_limit=length_lower[emission_time, LoS], upper_limit=length_upper[emission_time, LoS])

DensityPrior = ElectronDensityPrior(mean=DenMean[emission_time, LoS], sigma=DenErr[emission_time, LoS])

FulcherPrior = HotTempPrior(fulcherLimits[emission_time, LoS])

posterior = Posterior(
    components=[likelihood, HardLimit, ColdHotPrior, length_prior, DensityPrior, FulcherPrior]
)


"""
search for a good starting point
"""
rng = default_rng()
hypercube_samples = rng.uniform(low=HardLimit.lwr, high=HardLimit.upr, size=[100000, 6])
hypercube_probs = posterior(hypercube_samples)
grid_centre = hypercube_samples[hypercube_probs.argmax(), :]


"""
Run the algorithm
"""
# grid_spacing = array([0.15, 0.15, 0.15, 0.15, 0.1, 0.05])
# grid = PdfGrid(spacing=grid_spacing, offset=grid_centre, convergence=2e-2)

grid_spacing = array([0.05, 0.08, 0.08, 0.12, 0.05, 0.015])
grid_bounds = array([[log(0.2), log(0.2), log(5e18), log(1e-3), 0.01, 0.02], [log(5), log(1), log(4e19), log(1), 0.98, 0.4]]).T
grid = PdfGrid(spacing=grid_spacing, offset=grid_centre, bounds=grid_bounds, convergence=2e-2, n_samples=10000, n_climbs=200)

t1 = perf_counter()
while grid.state != "end":
    params = grid.get_parameters()
    P = posterior(params)
    grid.give_probabilities(P)
t2 = perf_counter()

print(f"\n # RUNTIME: {(t2-t1):.1f} s")

grid.plot_convergence()

adaptive_sample = grid.generate_sample(50000)
matrix_plot(adaptive_sample.T)
labels = ["ln_te_hot", "ln_te_cold", "ln_ne", "ln_ratio", "f_mol", "path_length"]

m = 17
hot_temp = linspace(log(0.2), log(5), m)
cold_temp = linspace(log(0.2), log(1), m)
e_density = linspace(log(5e18), log(4e19), m)
n_frac = linspace(log(1e-3), log(1), m)
mol_frac = linspace(0.01, 0.98, m)
path = linspace(0.02, 0.4, m)
params = (hot_temp, cold_temp, e_density, n_frac, mol_frac, path)

t1 = perf_counter()
evaluated_prob, marginals = EvaluatedGrid((hot_temp, cold_temp, e_density, n_frac, mol_frac, path), posterior)
t2 = perf_counter()

print(f"\n # RUNTIME: {(t2-t1):.1f} s")

labels = ["ln_te_hot", "ln_te_cold", "ln_ne", "ln_ratio", "f_mol", "path_length"]
cols = ["red", "blue", "green", "purple", "orange", "cyan"]
fig = plt.figure(figsize=(12, 7))
for i, lab in enumerate(labels):
    ax = fig.add_subplot(2, 3, i + 1)
    points, probs = grid.get_marginal([i])
    ax.plot(points, probs, ".-", color=cols[i])
    ax.plot(params[i], marginals[i], "--", color="black")
    ax.set_ylim([0., None])
    ax.set_title(lab)
    ax.grid()
plt.show()

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

# Resample if hard limit exceeded
selection_resample = np.all(np.logical_and(full_sample>HardLimit.lwr, full_sample<HardLimit.upr),axis=1)
while np.sum(selection_resample)>0:
    full_sample[selection_resample, :] = params[selection_resample, :] + rng.uniform(
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


alpha_gaussian_f, pdf_input_Da_f, full_kde_eval_a = Kernal(model, full_sample, 0, simulated_data[0])
beta_gaussian_f, pdf_input_Db_f, full_kde_eval_b = Kernal(model, full_sample, 1, simulated_data[1])

alpha_gaussian_a, pdf_input_Da_a, adap_kde_eval_a = Kernal(model, adaptive_sample, 0, simulated_data[0])
beta_gaussian_a, pdf_input_Db_a, adap_kde_eval_b = Kernal(model, adaptive_sample, 1, simulated_data[1])

cols = ["red", "blue"]
fig = plt.figure(figsize=(12, 7))

ax1 = fig.add_subplot(1, 2, 1)
# ax1.plot(alpha_gaussian_f, pdf_input_Da_f, color='black')
ax1.plot(alpha_gaussian_a, pdf_input_Da_a, color='black', linestyle='--', label='Input data')
ax1.plot(alpha_gaussian_f, full_kde_eval_a, color='red', label='Full Grid Output')
ax1.plot(alpha_gaussian_a, adap_kde_eval_a, color='blue', label='Adaptive Grid Output')
ax1.set_title("Simulated Balmer-Alpha Emission Intensity")
ax1.legend()
ax1.grid()

ax2 = fig.add_subplot(1, 2, 2)
# ax2.plot(gamma_gaussian_f, pdf_input_Dg_f, color='black')
ax2.plot(beta_gaussian_a, pdf_input_Db_a, color='black', linestyle='--', label='Input data')
ax2.plot(beta_gaussian_f, full_kde_eval_b, color='red', label='Full Grid Output')
ax2.plot(beta_gaussian_a, adap_kde_eval_b, color='blue', label='Adaptive Grid Output')
ax2.set_title("Simulated Balmer-Beta Emission Intensity")
ax2.legend()
ax2.grid()

plt.show()
