from numpy import array, log
from pdfgrid import PdfGrid
from inference.plotting import matrix_plot
from agsi.priors import *
from agsi.likelihood import *


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


"""
search for a good starting point
"""
from numpy.random import default_rng
rng = default_rng()
hypercube_samples = rng.uniform(low=HardLimit.lwr, high=HardLimit.upr, size=[100000, 6])
hypercube_probs = posterior(hypercube_samples)
grid_centre = hypercube_samples[hypercube_probs.argmax(), :]


"""
Run the algorithm
"""
grid_spacing = array([0.15, 0.15, 0.15, 0.15, 0.1, 0.05])
grid = PdfGrid(spacing=grid_spacing, offset=grid_centre, convergence=2e-2)

while grid.state != "end":
    params = grid.get_parameters()
    P = posterior(params)
    grid.give_probabilities(P)

grid.plot_convergence()

"""
Plot marginals
"""
sample = grid.generate_sample(50000)
labels = ["ln_te_hot", "ln_te_cold", "ln_ne", "ln_ratio", "f_mol", "path_length"]
matrix_plot(sample.T)

import matplotlib.pyplot as plt
cols = ["red", "blue", "green", "purple", "orange", "black"]
fig = plt.figure(figsize=(12, 7))
for i, lab in enumerate(labels):
    ax = fig.add_subplot(2, 3, i + 1)

    points, probs = grid.get_marginal([i])
    ax.plot(points, probs, ".-", color=cols[i])
    ax.set_title(lab)
    ax.grid()
plt.show()

from numpy import stack, array
probs = array(grid.probability)
inside_count = (probs > -1e5).sum()
print(inside_count, len(grid.coordinates), inside_count / len(grid.coordinates))

