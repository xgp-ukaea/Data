from numpy import array, linspace, zeros, exp, log, pi, sqrt
from pdfgrid import PdfGrid
from inference.plotting import matrix_plot
from agsi.priors import *
from agsi.likelihood import *

# The SpecData class is used to specify the band type and emission data for that band
# This class is then utilised for construction of PEC models and ultimately the production of a likelihood
EmissionData = SpecData(
    lines=["D_alpha", "D_gamma", "D_delta"],
    brightness=array([1e20, 1e19, 1e18]),
    uncertainty=array([1e18, 1e17, 1e16])
)

# SpecAnalysisLikelihood is a likelihood class that provides a result based upon the contents of SpecData
likelihood = SpecAnalysisLikelihood(
    measurements=EmissionData
)

# The BoundaryPrior class is used to define hard limits for all parameters
# The set-up of the Chris' classes requires the parameters to be input in the following order:
# T_e_hot, T_e_cold, n_e, n_frac, f_mol, path_length
# The function returns a large negative number if these boundaries are breached
HardLimit = BoundaryPrior(
    upper_limits=array([log(5), log(1), log(1e21), log(100), 0.99, 0.4]),
    lower_limits=array([log(0.2), log(0.02), log(1e17), log(1e-3), 0.01, 0.02])
)

# ColdTempPrior sets a hard limit on ensuring the recombinant temperature is less than the excitation temperature
# Also returns a large negative number
ColdHotPrior = ColdTempPrior()

# The Posterior class sums all the components used
posterior = Posterior(
    components=[likelihood, HardLimit, ColdHotPrior]
)

# Must be in same order as parameters in BoundaryPrior, to match the rest of Chris' classes
grid_spacing = array([0.5, 0.1, 2e19, 0.2, 0.2, 0.05])
grid_centre = array([2.5, 0.5, 1e20, 1, 0.5, 0.2])
grid = PdfGrid(spacing=grid_spacing, offset=grid_centre)

while grid.state != "end":
    GridTest = grid.get_parameters()
    P = posterior(GridTest)
    grid.give_probabilities(P)

sample = grid.generate_sample(10000)
matrix_plot(sample.T)
