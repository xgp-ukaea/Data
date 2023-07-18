from numpy import array, log, load, exp, linspace
from pdfgrid import PdfGrid
from agsi.priors import *
from agsi.likelihood import *
from numpy.random import default_rng
import dms.analysis.emission.Balmer_analysis as BA
import matplotlib.pyplot as plt
import scipy.stats as scis

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


def PosteriorSample(D_alpha, D_beta, uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits):
    simulated_data = array([D_alpha, D_beta])
    EmissionData = SpecData(
        lines=["D_alpha", "D_beta"],
        brightness=simulated_data,
        uncertainty=simulated_data*uncertainty
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

    length_prior = PathLengthPrior(mode=length_mode, lower_limit=length_lower, upper_limit=length_upper)

    FulcherPrior = HotTempPrior(fulcherLimits)

    DensityPrior = ElectronDensityPrior(mean=DenMean, sigma=DenErr)

    posterior = Posterior(
        components=[likelihood, HardLimit, ColdHotPrior, length_prior, FulcherPrior, DensityPrior]
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

    while grid.state != "end":
        params = grid.get_parameters()
        P = posterior(params)
        grid.give_probabilities(P)
    """
    labels = ["ln_te_hot", "ln_te_cold", "ln_ne", "ln_ratio", "f_mol", "path_length"]
    cols = ["red", "blue", "green", "purple", "orange", "cyan"]
    fig = plt.figure(figsize=(12, 7))
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(2, 3, i + 1)
        points, probs = grid.get_marginal([i])
        ax.plot(points, probs, ".-", color=cols[i])
        ax.set_ylim([0., None])
        ax.set_title(lab)
        ax.grid()
    plt.show()
    """
    adaptive_sample = grid.generate_sample(50000)

    return adaptive_sample


def PressureKernel(adaptive_sample):
    pressure_sample = exp(adaptive_sample[:, 0]) * exp(adaptive_sample[:, 2])
    kde = scis.gaussian_kde(pressure_sample)
    xmin, xmax = min(pressure_sample), max(pressure_sample)
    x = linspace(xmin, xmax, 10000)
    kde_eval = kde(x)
    kde_eval = kde_eval / sum(kde_eval)

    plt.plot(x, kde_eval)
    plt.show()

    return x, kde_eval


def PressureCheck(sample):
    pressure_sample = exp(sample[:, 0]) * exp(sample[:, 2])

    if max(pressure_sample) >= 2e20:
        return max(pressure_sample)
