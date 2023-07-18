from agsi.analysisOld import ImportData, PosteriorSample#, PressureKernel, PressureCheck
import numpy as np


D_alpha, D_beta, uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits = ImportData("input.npy")

Iter = 50000

DenMC = np.zeros([np.shape(DenMean)[0], np.shape(DenMean)[1],Iter]) + np.nan
TeEMC = np.zeros([np.shape(DenMean)[0], np.shape(DenMean)[1],Iter]) + np.nan
TeRMC = np.zeros([np.shape(DenMean)[0], np.shape(DenMean)[1],Iter]) + np.nan
DLMC = np.zeros([np.shape(DenMean)[0], np.shape(DenMean)[1],Iter]) + np.nan
noneMC = np.zeros([np.shape(DenMean)[0], np.shape(DenMean)[1],Iter]) + np.nan
fmolMC = np.zeros([np.shape(DenMean)[0], np.shape(DenMean)[1],Iter]) + np.nan

# for i in range(1):
for i in range(len(D_alpha[:, 0])):
    # for j in range(1):
    for j in range(len(D_alpha[0, :])):
        sample = PosteriorSample(D_alpha[i, j], D_beta[i, j], uncertainty, length_mode[i, j], length_lower[i, j], length_upper[i, j], DenMean[i, j], DenErr[i, j], fulcherLimits[i, j])

        TeEMC[i, j, :] = np.exp(sample[:, 0])
        TeRMC[i, j, :] = np.exp(sample[:, 1])
        DenMC[i, j, :] = np.exp(sample[:, 2])
        noneMC[i, j, :] = np.exp(sample[:, 3])
        fmolMC[i, j, :] = sample[:, 4]
        DLMC[i, j, :] = sample[:, 5]

        # x, kde_eval = PressureKernel(sample)


input = np.load('input.npy', allow_pickle=True)
output = dict()
output['input'] = input
output['ResultMC'] = dict()
output['ResultMC']['DenMC'] = DenMC
output['ResultMC']['TeEMC'] = TeEMC
output['ResultMC']['TeRMC'] = TeRMC
output['ResultMC']['fmolMC'] = fmolMC
output['ResultMC']['DLMC'] = DLMC
output['ResultMC']['noneMC'] = noneMC

np.save('output.npy', output)

y = 0