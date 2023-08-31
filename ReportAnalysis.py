from numpy import array, log, load, shape, nan, unravel_index
from pdfgrid import PdfGrid
from agsi.priors import *
from agsi.likelihood import *
from numpy.random import default_rng
import dms.analysis.emission.Balmer_analysis as BA
from multiprocessing import Pool
import numpy as np


def ImportData(input_data, band_check):
    Uncertainty = input_data['AbsErr']
    length_mode = input_data['DL']
    length_lower = input_data['DLL']
    length_upper = input_data['DLH']
    fulcher = input_data['Fulcher']
    fulcherLimits, te_lim_low, te_lim_high = BA.get_fulcher_constraint_spline(fulcher, telim=True, cummax=True)
    fulcherLimits = fulcherLimits
    DenMean = input_data['Den']
    DenErr = input_data['DenErr']
    n1 = int(input_data['n1'])
    n2 = int(input_data['n2'])

    if band_check == 3:
        D_alpha = input_data['n1Int']
        n2_line = input_data['n2Int']

        return D_alpha, n2_line, Uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits, n1, n2

    else:
        D_alpha = input_data['DaMea']
        n1_line = input_data['n1Int']
        n2_line = input_data['n2Int']

        return D_alpha, n1_line, n2_line, Uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits, n1, n2


def SpecificBounds(input_data):
    DenMax = input_data['DenMax']
    DenMin = input_data['DenMin']
    NeutralMax = input_data['noneH']
    NeutralMin = input_data['noneL']

    return DenMax, DenMin, NeutralMax, NeutralMin


def EmissionLines(n1, n2):
    n_lines = ["D_alpha", "D_beta", "D_gamma", "D_delta", "D_epsilon", "D_zeta", "D_eta"]
    if n1 == 3:
        line = n_lines[n1 - 3], n_lines[n2 - 3]
    else:
        line = n_lines[0], n_lines[n1 - 3], n_lines[n2 - 3]

    return line


def TwoBands(D_alpha, n2_line, uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits, n1, n2, DenMax=4e19, DenMin=5e18, NeutralMax=1, NeutralMin=1e-3):

    if np.isnan(n2_line):
        adaptive_sample = np.full((500, 6), np.nan)

    else:
        lines = EmissionLines(n1, n2)
        simulated_data = array([D_alpha, n2_line])
        EmissionData = SpecData(
            lines=[lines[0], lines[1]],
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
            upper_limits=array([log(5), log(1), log(DenMax), log(NeutralMax), 0.95, 0.4]),
            lower_limits=array([log(0.2), log(0.02), log(DenMin), log(NeutralMin), 0.01, 0.02])
        )

        ColdHotPrior = ColdTempPrior()

        length_prior = PathLengthPrior(mode=length_mode, lower_limit=length_lower, upper_limit=length_upper)

        FulcherPrior = HotTempPrior(fulcherLimits)

        DensityPrior = ElectronDensityPrior(mean=DenMean, sigma=DenErr)

        posterior = Posterior(
            components=[likelihood, ColdHotPrior, length_prior, FulcherPrior, DensityPrior],
            boundary_prior=HardLimit
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
        grid_spacing = array([0.06, 0.09, 0.09, 0.15, 0.06, 0.02])
        grid_bounds = array([[log(0.2), log(0.02), log(DenMin), log(NeutralMin), 0.01, 0.02], [log(5), log(1), log(DenMax), log(NeutralMax), 0.95, 0.4]]).T
        grid = PdfGrid(spacing=grid_spacing, offset=grid_centre, bounds=grid_bounds, convergence=1e-1, n_samples=10000, n_climbs=200)

        while grid.state != "end":
            params = grid.get_parameters()
            P = posterior(params)
            grid.give_probabilities(P)

        adaptive_sample = grid.generate_sample(HardLimit, 500)

    return adaptive_sample


def ThreeBands(D_alpha, n1_line, n2_line, uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits, n1, n2, DenMax=4e19, DenMin=5e18, NeutralMax=1, NeutralMin=1e-3):

    if np.isnan(n1_line) or np.isnan(n2_line):
        adaptive_sample = np.full((500, 6), np.nan)

    else:
        """
        Construct emission model
        """
        lines = EmissionLines(n1, n2)
        emission_data = array([D_alpha, n1_line, n2_line])
        EmissionData = SpecData(
            lines=[lines[0], lines[1], lines[2]],
            brightness=emission_data,
            uncertainty=emission_data*uncertainty
        )

        """
        Build posterior
        """
        likelihood = SpecAnalysisLikelihood(
            measurements=EmissionData
        )

        HardLimit = BoundaryPrior(
            upper_limits=array([log(5), log(1), log(DenMax), log(NeutralMax), 0.95, 0.4]),
            lower_limits=array([log(0.2), log(0.02), log(DenMin), log(NeutralMin), 0.01, 0.02])
        )

        ColdHotPrior = ColdTempPrior()

        length_prior = PathLengthPrior(mode=length_mode, lower_limit=length_lower, upper_limit=length_upper)

        FulcherPrior = HotTempPrior(fulcherLimits)

        DensityPrior = ElectronDensityPrior(mean=DenMean, sigma=DenErr)

        posterior = Posterior(
            components=[likelihood, ColdHotPrior, length_prior, FulcherPrior, DensityPrior],
            boundary_prior=HardLimit
        )

        """
        Run adaptive grid algorithm
        """
        grid_spacing = array([0.06, 0.09, 0.09, 0.15, 0.06, 0.02])
        grid_bounds = array([[log(0.2), log(0.02), log(DenMin), log(NeutralMin), 0.01, 0.02], [log(5), log(1), log(DenMax), log(NeutralMax), 0.95, 0.4]]).T
        grid_centre = (grid_bounds[:, 1] - grid_bounds[:, 0]) / 2
        grid = PdfGrid(spacing=grid_spacing, offset=grid_centre, bounds=grid_bounds, convergence=1e-1, n_samples=10000, n_climbs=200)

        while grid.state != "end":
            params = grid.get_parameters()
            P = posterior(params)
            grid.give_probabilities(P)

        adaptive_sample = grid.generate_sample(HardLimit, 500)

    return adaptive_sample


def OutputStructure(input, TeEMC, TeRMC, DenMC, noneMC, fmolMC, DLMC, PresMC):
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
    output['ResultMC']['PresMC'] = PresMC

    return output


def FullAnalysis(input_file, poolcount=48, specific_bound=False):
    input_data = load(input_file, allow_pickle=True)
    input_data = input_data[()]
    band_check = input_data['n1']

    Iter = 500

    p = []
    pool = Pool(poolcount)

    if band_check == 3:
        D_alpha, n2_line, Uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits, n1, n2 = ImportData(input_data, band_check)

        DenMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        TeEMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        TeRMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        DLMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        noneMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        fmolMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        PresMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan

        S = shape(DenMean)

        print('appending operations to pool')

        if specific_bound:
            DenMax, DenMin, NeutralMax, NeutralMin = SpecificBounds(input_data)

            for l in range(0, S[0]*S[1]):
                i, j = unravel_index(l, S)

                p.append(pool.apply_async(TwoBands, (D_alpha[i, j], n2_line[i, j], Uncertainty, length_mode[i, j], length_lower[i, j], length_upper[i, j], DenMean[i, j], DenErr[i, j], fulcherLimits[i, j], n1, n2), dict(DenMax=DenMax, DenMin=DenMin, NeutralMax=NeutralMax[i, j], NeutralMin=NeutralMin[i, j])))

        else:
            for l in range(0, S[0] * S[1]):
                i, j = unravel_index(l, S)

                p.append(pool.apply_async(TwoBands, (D_alpha[i, j], n2_line[i, j], Uncertainty, length_mode[i, j], length_lower[i, j], length_upper[i, j], DenMean[i, j], DenErr[i, j], fulcherLimits[i, j], n1, n2)))

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
            PresMC[i, j, :] = exp(sample[l][:, 0]+sample[l][:,2])

        output = OutputStructure(input_data, TeEMC, TeRMC, DenMC, noneMC, fmolMC, DLMC, PresMC)

    else:
        D_alpha, n1_line, n2_line, Uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits, n1, n2 = ImportData(input_data, band_check)

        DenMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        TeEMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        TeRMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        DLMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        noneMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        fmolMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan
        PresMC = zeros([shape(DenMean)[0], shape(DenMean)[1], Iter]) + nan

        S = shape(DenMean)

        print('appending operations to pool')

        if specific_bound:
            DenMax, DenMin, NeutralMax, NeutralMin = SpecificBounds(input_data)

            for l in range(0, S[0] * S[1]):
                i, j = unravel_index(l, S)

                p.append(pool.apply_async(ThreeBands, (D_alpha[i, j], n1_line[i, j], n2_line[i, j], Uncertainty, length_mode[i, j], length_lower[i, j], length_upper[i, j], DenMean[i, j], DenErr[i, j], fulcherLimits[i, j], n1, n2), dict(DenMax=DenMax, DenMin=DenMin, NeutralMax=NeutralMax[i, j], NeutralMin=NeutralMin[i, j])))

        else:
            for l in range(0, S[0] * S[1]):
                i, j = unravel_index(l, S)

                p.append(pool.apply_async(ThreeBands, (D_alpha[i, j], n1_line[i, j], n2_line[i, j], Uncertainty, length_mode[i, j], length_lower[i, j], length_upper[i, j], DenMean[i, j], DenErr[i, j], fulcherLimits[i, j], n1, n2)))

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
            PresMC[i, j, :] = exp(sample[l][:, 0]+sample[l][:, 2])

        output = OutputStructure(input_data, TeEMC, TeRMC, DenMC, noneMC, fmolMC, DLMC, PresMC)

    return output

