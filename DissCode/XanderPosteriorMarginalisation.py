import matplotlib.pyplot as plt
from numpy import array, linspace, zeros, exp, log, pi, sqrt


# function for a straight line
def model(x, theta):
    # theta is the vector of model parameters
    m, c = theta
    return m*x + c


# function for calculating and returning the gaussian log-likelihood
def log_likelihood(x, y, y_err, theta):
    z = (y - model(x, theta)) / y_err
    logl = - log(y_err*(sqrt(2*pi))).sum() - 0.5*(z**2).sum()
    return logl


# dataset we're analysing
x = array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
y = array([1.457, 2.466, 4.822, 4.33, 5.96, 6.222, 5.081, 7.575, 6.812, 7.426])
y_err = array([0.7, 0.907, 1.066, 1.2, 1.318, 1.425, 1.523, 1.614, 1.7, 1.781])


# plotting the data
# plt.errorbar(x, y, yerr=y_err, ls="none", marker="o", markerfacecolor="none")
# plt.tight_layout()
# plt.grid()
# plt.show()


# m defines the number of intervals for each parameter
m = 128
gradient_axis = linspace(0, 1.5, m)
offset_axis = linspace(-2, 5, m)
likelihood_2D = zeros([m, m])


# loop calculating the log-likelihood for all gradient and offset combinations
for i in range(m):
    for j in range(m):
        theta = [gradient_axis[i], offset_axis[j]]
        likelihood_2D[i, j] = log_likelihood(x, y, y_err, theta)
# returning the log-likelihood into the likelihood
likelihood_2D = exp(likelihood_2D)


# the posterior can be calculated by multiplying the likelihood by a prior
# exponential prior on the gradient exp(-x / t) with t = 0.5
# broadcasting needed to only multiply gradient dimension of likelihood as
# the prior only affects this dimension
t = 0.5
posterior_2D = likelihood_2D * exp(-gradient_axis / t)[:, None]


#plt.contourf(gradient_axis, offset_axis, posterior_2D.T)
#plt.colorbar(label = 'Posterior')
#plt.xlabel('Gradient')
#plt.ylabel('Intercept')
#plt.show()


# integrating each dimension can be performed be summing along the individual dimension
# to obtain the marginal distribution
gradient_marg = posterior_2D.sum(axis=1)
offset_marg = posterior_2D.sum(axis=0)


plt.plot(gradient_axis, gradient_marg, label='Gradient')
plt.plot(offset_axis, offset_marg, label='Offset')
plt.xlabel('gradient, offset')
plt.ylabel('Probability density')
plt.legend()
plt.show()