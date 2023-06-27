from numpy import array, log, linspace, unravel_index, meshgrid, sum, stack
import matplotlib.pyplot as plt
from inference.plotting import matrix_plot
from agsi.priors import *
from agsi.likelihood import *
from numpy.random import choice
from scipy.integrate import simps


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

posterior_6D = zeros([m, m, m, m, m, m])

# Generate parameter combinations using meshgrid
hot_temp_grid, cold_temp_grid, e_density_grid, n_frac_grid, mol_frac_grid, path_grid = meshgrid(
    hot_temp, cold_temp, e_density, n_frac, mol_frac, path, indexing='ij'
)

# Reshape the grids to obtain a 2D array of parameter combinations
param_combinations = array([
    hot_temp_grid.flatten(),
    cold_temp_grid.flatten(),
    e_density_grid.flatten(),
    n_frac_grid.flatten(),
    mol_frac_grid.flatten(),
    path_grid.flatten()
]).T

# Compute posterior values for all parameter combinations
posterior_6D = posterior(param_combinations.reshape(-1, 6))

# Reshape the posterior array to match the original shape
posterior_6D = posterior_6D.reshape((m, m, m, m, m, m))

# Converts the posterior to 1-dimension and normalises
p = posterior_6D.flatten()
p = exp(p - p.max())
p /= p.sum()

# Taking samples from the full grid based on probability weighting
index = choice(len(p), size=10000, p=p)

# Breaks the sample number down into the combination of the 6 parameters creating that cell
sample = unravel_index(index, (m, m, m, m, m, m))

matrix_plot(sample)

# Marginalise by summing the dimensions
marginal_hot_temp = posterior_6D.sum(axis=(1, 2, 3, 4, 5))
marginal_cold_temp = posterior_6D.sum(axis=(0, 2, 3, 4, 5))
marginal_e_density = posterior_6D.sum(axis=(0, 1, 3, 4, 5))
marginal_n_frac = posterior_6D.sum(axis=(0, 1, 2, 4, 5))
marginal_mol_frac = posterior_6D.sum(axis=(0, 1, 2, 3, 5))
marginal_path = posterior_6D.sum(axis=(0, 1, 2, 3, 4))

# Normalise the marginal distributions
marginal_hot_temp /= simps(marginal_hot_temp, x=hot_temp)
marginal_cold_temp /= simps(marginal_cold_temp, x=cold_temp)
marginal_e_density /= simps(marginal_e_density, x=e_density)
marginal_n_frac /= simps(marginal_n_frac, x=n_frac)
marginal_mol_frac /= simps(marginal_mol_frac, x=mol_frac)
marginal_path /= simps(marginal_path, x=path)

# Plotting the marginal distributions
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].plot(exp(hot_temp), marginal_hot_temp)
axes[0, 0].set_xlabel('Hot Temp')
axes[0, 1].plot(exp(cold_temp), marginal_cold_temp)
axes[0, 1].set_xlabel('Cold Temp')
axes[0, 2].plot(exp(e_density), marginal_e_density)
axes[0, 2].set_xlabel('Electron Density')
axes[1, 0].plot(exp(n_frac), marginal_n_frac)
axes[1, 0].set_xlabel('N Fraction')
axes[1, 1].plot(mol_frac, marginal_mol_frac)
axes[1, 1].set_xlabel('Mol Fraction')
axes[1, 2].plot(path, marginal_path)
axes[1, 2].set_xlabel('Path')
plt.tight_layout()
plt.show()

