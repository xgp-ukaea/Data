from agsi.analysisOld import ImportData, PosteriorSample


D_alpha, D_beta, uncertainty, length_mode, length_lower, length_upper, DenMean, DenErr, fulcherLimits = ImportData("input.npy")

# for i in range(1):
for i in range(len(D_alpha[:, 0])):
    # for j in range(1):
    for j in range(len(D_alpha[0, :])):
        sample = PosteriorSample(D_alpha[i, j], D_beta[i, j], uncertainty, length_mode[i, j], length_lower[i, j], length_upper[i, j], DenMean[i, j], DenErr[i, j], fulcherLimits[i, j])

y = 0