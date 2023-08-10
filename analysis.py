from numpy import array, log, load, exp, linspace, ones, zeros, shape, nan, unravel_index
from pdfgrid import PdfGrid
from agsi.priors import *
from agsi.likelihood import *
from numpy.random import default_rng
import dms.analysis.emission.Balmer_analysis as BA
import matplotlib.pyplot as plt
import scipy.stats as scis
from multiprocessing import Pool


def ImportData(input_data, band_check):
    Uncertainty = input_data['AbsErr']
    length_mode = input_data['DL']
    length_lower = input_data['DLL']
    length_upper = input_data['DLH']
    fulcher = input_data['Fulcher']
    fulcherLimits, te_lim_low, te_lim_high = BA.get_fulcher_constraint_spline(fulcher, telim=True)
    fulcherLimits = fulcherLimits
    DenMean = input_data['Den']
    DenErr = input_data['DenErr']
    band_check = input_data['n1']

    if band_check == 3:
        D_alpha = input_data['n1Int']
        D_beta = input_data['n2Int']
        input_data = 0

        return D_alpha, D_beta, Uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits

    else:
        D_gamma = input_data['n1Int']
        D_delta = input_data['n2Int']
        D_alpha = input_data['DaMean'][:, None] * ones((1, len(D_delta[1])))
        input_data = 0

        return D_alpha, D_gamma, D_delta, Uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits


def AlphaBeta(D_alpha, D_beta, uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits):

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
    adaptive_sample = grid.generate_sample(500)

    return adaptive_sample


def AlphaGammaDelta(D_alpha, D_gamma, D_delta, uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits):
    simulated_data = array([D_alpha, D_gamma, D_delta])
    EmissionData = SpecData(
        lines=["D_alpha", "D_gamma", "D_delta"],
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
    adaptive_sample = grid.generate_sample(500)

    return adaptive_sample


def OutputStructure(input, TeEMC, TeRMC, DenMC, noneMC, fmolMC, DLMC):
    input['n1'] = int(input['n1'])
    input['n2'] = int(input['n2'])
    output = dict()
    output['input'] = input
    output['ResultMC'] = dict()
    output['ResultMC']['DenMC'] = DenMC
    output['ResultMC']['TeEMC'] = TeEMC
    output['ResultMC']['TeRMC'] = TeRMC
    output['ResultMC']['fmolMC'] = fmolMC
    output['ResultMC']['DLMC'] = DLMC
    output['ResultMC']['noneMC'] = noneMC

    return output


def FullAnalysis(input_file, poolcount=4):
    input_data = load(input_file, allow_pickle=True)
    input_data = input_data[()]
    band_check = input_data['n1']

    Iter = 500

    p = []
    pool = Pool(poolcount)

    if band_check == 3:
        D_alpha, D_beta, Uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits = ImportData(input_data, band_check)

        DenMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        TeEMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        TeRMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        DLMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        noneMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        fmolMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan

        S = shape(DenMean)

        print('appending operations to pool')
        for l in range(0, S[0]*S[1]):

            i, j = unravel_index(l, S)

            p.append(pool.apply_async(AlphaBeta, (D_alpha[i, j], D_beta[i, j], Uncertainty, length_mode[i, j], length_lower[i, j], length_upper[i, j], DenMean[i, j], DenErr[i, j], fulcherLimits[i, j])))

        print('executing pool')
        sample = [p[hh].get() for hh in range(len(p))]

        for l in range(0, S[0]*S[1]):
            i, j = unravel_index(l, S)

            TeEMC[i, j, :] = exp(sample[l][:, 0])
            TeRMC[i, j, :] = exp(sample[l][:, 1])
            DenMC[i, j, :] = exp(sample[l][:, 2])
            noneMC[i, j, :] = exp(sample[l][:, 3])
            fmolMC[i, j, :] = sample[l][:, 4]
            DLMC[i, j, :] = sample[l][:, 5]

        output = OutputStructure(input_data, TeEMC, TeRMC, DenMC, noneMC, fmolMC, DLMC)

    else:
        D_alpha, D_gamma, D_delta, Uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits = ImportData(input_data, band_check)

        DenMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        TeEMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        TeRMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        DLMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        noneMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        fmolMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan

        S = shape(DenMean)

        print('appending operations to pool')
        for l in range(0, S[0] * S[1]):
            i, j = unravel_index(l, S)

            p.append(pool.apply_async(AlphaGammaDelta, (D_alpha[i, j], D_gamma[i, j], D_delta[i, j], Uncertainty, length_mode[i, j], length_lower[i, j], length_upper[i, j], DenMean[i, j], DenErr[i, j], fulcherLimits[i, j])))

        print('executing pool')
        sample = [p[hh].get() for hh in range(len(p))]

        for l in range(0, S[0] * S[1]):
            i, j = unravel_index(l, S)

            TeEMC[i, j, :] = exp(sample[l][:, 0])
            TeRMC[i, j, :] = exp(sample[l][:, 1])
            DenMC[i, j, :] = exp(sample[l][:, 2])
            noneMC[i, j, :] = exp(sample[l][:, 3])
            fmolMC[i, j, :] = sample[l][:, 4]
            DLMC[i, j, :] = sample[l][:, 5]

        output = OutputStructure(input_data, TeEMC, TeRMC, DenMC, noneMC, fmolMC, DLMC)

    return output

