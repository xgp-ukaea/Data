import matplotlib.pyplot as plt
from numpy import array, linspace, zeros, exp, log, pi, sqrt
from scipy.integrate import simps


class Posterior:
    def __init__(self, x, y, y_err):
        # defining the parameters inside the class
        self.x = x
        self.y = y
        self.y_err = y_err
        self.t = 0.5

    # defining what is returned when the class is called
    def __call__(self, theta):
        return self.likelihood(theta) + self.prior(theta)

    def likelihood(self, theta):
        z = (self.y - self.model(self.x, theta)) / self.y_err
        logl = - log(self.y_err * (sqrt(2 * pi))).sum() - 0.5 * (z ** 2).sum()
        return logl

    def prior(self, theta):
        gradient, offset = theta
        return exp(-gradient / self.t)

    @staticmethod
    def model(x, theta):
        m, c = theta
        return m*x + c



# dataset we're analysing
x = array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
y = array([1.457, 2.466, 4.822, 4.33, 5.96, 6.222, 5.081, 7.575, 6.812, 7.426])
y_err = array([0.7, 0.907, 1.066, 1.2, 1.318, 1.425, 1.523, 1.614, 1.7, 1.781])

# feeding the input parameters into the class
posterior = Posterior(
    x=x,
    y=y,
    y_err=y_err
)

m = 128
gradient_axis = linspace(0, 1.5, m)
offset_axis = linspace(-2, 5, m)
posterior_2D = zeros([m, m])

for i in range(m):
    for j in range(m):
        theta = [gradient_axis[i], offset_axis[j]]
        posterior_2D[i, j] = posterior(theta)
posterior_2D = exp(posterior_2D)


gradient_marg = posterior_2D.sum(axis=1)
offset_marg = posterior_2D.sum(axis=0)

# normalise the parameters
gradient_marg /= simps(gradient_marg, x=gradient_axis)
offset_marg /= simps(offset_marg, x=offset_axis)

plt.plot(gradient_axis, gradient_marg, label='Gradient')
plt.plot(offset_axis, offset_marg, label='Offset')
plt.xlabel('Gradient, Offset')
plt.ylabel('Probability density')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
