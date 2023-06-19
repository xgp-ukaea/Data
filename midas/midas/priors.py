
from numpy import array, exp, zeros, maximum, minimum, reciprocal, heaviside, log, subtract, mean, eye, diag
from numpy import sum as npsum
from scipy.linalg import solve_triangular, solve_banded, ldl
from scipy.sparse import diags


class StaticPressurePrior(object):
    def __init__(self, plasma = None, max_pressure = 7e20, sigma = 7e19):
        self.plasma = plasma
        self.P_max = max_pressure
        self.variance = sigma**2

        self.parameters = ['Te', 'ne']
        self.parameter_types = ['field', 'field']

    def __call__(self):
        return self.log_prior(self.plasma.get('Te'), self.plasma.get('ne'))

    def log_prior(self, Te, ne):
        dP = Te*ne - self.P_max
        return -0.5*npsum(maximum(dP,0.)**2)/self.variance

    def gradient(self):
        Te = self.plasma.get('Te')
        ne = self.plasma.get('ne')
        dP = Te*ne - self.P_max
        base = -maximum(dP, 0.) / self.variance

        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices['Te']] = base*ne # temperature gradients
        grad[self.plasma.slices['ne']] = base*Te # density gradients
        return grad






class BoundaryConditionPrior(object):
    def __init__(self, plasma = None, fields = ('Te', 'ne', 'n0'), boundary_vals = (0.75, 1e17, 3e16), boundary_indices = None):
        self.plasma = plasma
        self.inds = array(boundary_indices)

        self.parameters = fields
        self.parameter_types = ['field']*len(self.parameters)

        self.limits = boundary_vals
        self.variances = [(0.1*v)**2 for v in self.limits]

    def __call__(self):
        return sum( self.log_prior(*t) for t in zip(self.parameters, self.limits, self.variances))

    def log_prior(self, field, limit, variance):
        return -0.5*npsum(maximum(self.plasma.get(field)[self.inds] - limit, 0.)**2) / variance

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        for f, l, v in zip(self.parameters, self.limits, self.variances):
            grad[self.plasma.slices[f]][self.inds] = -maximum(self.plasma.get(f)[self.inds] - l,0.) / v
        return grad






class TemperatureDecayPrior(object):
    def __init__(self, plasma = None, sigma = 1.0):
        self.plasma = plasma
        self.variance = sigma**2

        # settings
        min_temp = 3.
        max_temp = 40.
        centre = 1.3
        scale = 0.075

        R = array( [v[0] for v in self.plasma.mesh.vertices] )
        arg = (R - centre) / scale
        self.Te_max = (max_temp - min_temp)/(1. + exp(arg)) + min_temp

        self.parameters = ['Te']
        self.parameter_types = ['field']

    def __call__(self):
        return self.log_prior(self.plasma.get('Te'))

    def log_prior(self, Te):
        dP = Te - self.Te_max
        return -0.5*npsum(maximum(dP,0.)**2) / self.variance

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        dP = self.plasma.get('Te') - self.Te_max
        grad[self.plasma.slices['Te']] = -maximum(dP,0.) / self.variance # temperature gradients
        return grad






class NeutralFractionPrior(object):
    def __init__(self, plasma = None, lam = 0.2, sigma = 0.1, b = 0.035):
        self.plasma = plasma
        self.lam = lam
        self.b = b
        self.variance = sigma**2

        self.parameters = ['Te', 'ne', 'n0']
        self.parameter_types = ['field']*3

    def __call__(self):
        return self.log_prior(self.plasma.get('Te'), self.plasma.get('ne'), self.plasma.get('n0'))

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        dTe, dne, dn0 = self.log_prior_gradient(self.plasma.get('Te'), self.plasma.get('ne'), self.plasma.get('n0'))
        grad[self.plasma.slices['Te']] = dTe # temperature gradients
        grad[self.plasma.slices['ne']] = dne # density gradients
        grad[self.plasma.slices['n0']] = dn0 # neutral gradients
        return grad

    def log_prior(self, Te, ne, n0):
        f = n0 / (n0 + ne)
        f_lim = (1-self.b)*exp(-self.lam*Te) + self.b
        dP = f/f_lim - 1
        return -0.5*npsum(maximum(dP,0.)**2)/self.variance

    def log_prior_gradient(self, Te, ne, n0):
        inv_conc = reciprocal(n0 + ne)
        f = n0 * inv_conc
        f_lim = (1 - self.b)*exp(-self.lam*Te) + self.b

        R = f/f_lim
        base = -R*maximum(R-1, 0.) / self.variance

        dTe = self.lam*((f_lim-self.b)/f_lim)*base
        dne = -base*inv_conc
        dn0 = (ne*base*inv_conc) / n0
        return dTe, dne, dn0






class DensityRatioPrior(object):
    def __init__(self, plasma = None, lam = 0.3, sigma = 0.1, a = 4., b = 0.05):
        self.plasma = plasma
        self.lam = lam
        self.a = a
        self.b = b
        self.variance = sigma**2

        self.parameters = ['Te', 'ne', 'n0']
        self.parameter_types = ['field']*3

    def __call__(self):
        return self.log_prior(self.plasma.get('Te'), self.plasma.get('ne'), self.plasma.get('n0'))

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        dTe, dne, dn0 = self.log_prior_gradient(self.plasma.get('Te'), self.plasma.get('ne'), self.plasma.get('n0'))
        grad[self.plasma.slices['Te']] = dTe # temperature gradients
        grad[self.plasma.slices['ne']] = dne # density gradients
        grad[self.plasma.slices['n0']] = dn0 # neutral gradients
        return grad

    def log_prior(self, Te, ne, n0):
        r = n0 / ne
        r_lim = self.a*exp(-self.lam*Te) + self.b
        dP = r/r_lim - 1
        return -0.5*npsum(maximum(dP,0.)**2)/self.variance

    def log_prior_gradient(self, Te, ne, n0):
        inv_ne = reciprocal(ne)
        r = n0 * inv_ne
        r_lim = self.a*exp(-self.lam*Te) + self.b

        R = r/r_lim
        base = -R*maximum(R-1, 0.) / self.variance

        dTe = self.lam*((r_lim-self.b)/r_lim)*base
        dne = -base*inv_ne
        dn0 = base / n0
        return dTe, dne, dn0






class MaxValuePrior(object):
    def __init__(self, plasma = None, field = 'Te', max_val = 45., uncertainty = 1.):

        self.plasma = plasma
        self.field = field
        self.parameters = [field]
        self.parameter_types = ['field']*len(self.parameters)

        self.limit = max_val
        self.variance = uncertainty**2

    def __call__(self):
        return self.log_prior(self.plasma.get(self.field))

    def gradient(self):
        deriv = self.log_prior_gradient(self.plasma.get(self.field))
        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices[self.field]] = deriv
        return grad

    def log_prior(self, field):
        return -0.5*npsum(maximum(field-self.limit,0.)**2)/self.variance

    def log_prior_gradient(self, field):
        return -npsum(maximum(field-self.limit,0.))/self.variance






class MaxLogGradientPrior(object):
    def __init__(self, plasma = None, min_scale_length = 0.02, uncertainty = 0.1):
        self.plasma = plasma
        self.G = plasma.mesh.edge_gradient_matrix()
        self.max_grad_sq = 1/min_scale_length**2
        self.variance = uncertainty**2

        self.parameters = ['Te', 'ne', 'n0']
        self.parameter_types = ['field']*3

    def __call__(self):
        Te_prior = self.log_prior(self.plasma.get('Te'))
        ne_prior = self.log_prior(self.plasma.get('ne'))
        n0_prior = self.log_prior(self.plasma.get('n0'))
        return Te_prior + ne_prior + n0_prior

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices['Te']] = self.log_prior_gradient(self.plasma.get('Te')) / self.plasma.get('Te')
        grad[self.plasma.slices['ne']] = self.log_prior_gradient(self.plasma.get('ne')) / self.plasma.get('ne')
        grad[self.plasma.slices['n0']] = self.log_prior_gradient(self.plasma.get('n0')) / self.plasma.get('n0')
        return grad

    def log_prior(self, field):
        grad = self.G.dot(log(field))
        y = grad**2 - self.max_grad_sq
        return -maximum(y,0).sum() / (self.max_grad_sq*self.variance)

    def log_prior_gradient(self, field):
        grad = self.G.dot(log(field))
        y = heaviside(grad**2 - self.max_grad_sq, 1.) * grad
        return -2*((self.G.T).dot(y)) / (self.max_grad_sq*self.variance)






def tridiagonal_banded_form(A):
    B = zeros([3, A.shape[0]])
    B[0, 1:] = diag(A, k = 1)
    B[1, :] = diag(A)
    B[2,:-1] = diag(A, k = -1)
    return B






class SymmetricSystemSolver(object):
    def __init__(self, A):
        L_perm, D, fwd_perms = ldl(A)

        self.L = L_perm[fwd_perms,:]
        self.DB = tridiagonal_banded_form(D)
        self.fwd_perms = fwd_perms
        self.rev_perms = fwd_perms.argsort()

    def __call__(self, b):
        # first store the permuted form of b
        b_perm = b[self.fwd_perms]

        # solve the system by substitution
        y = solve_triangular(self.L, b_perm, lower = True)
        h = solve_banded((1, 1), self.DB, y)
        x_perm = solve_triangular(self.L.T, h, lower = False)

        # now un-permute the solution vector
        x_sol = x_perm[self.rev_perms]
        return x_sol






class GaussianProcessPrior(object):
    def __init__(self, plasma = None, A = 3., L = 0.01):
        self.plasma = plasma
        self.A = A
        self.L = L

        self.mu_Te = log(0.2)
        self.mu_ne = log(1e17)
        self.mu_n0 = log(1e16)

        R_vals = [ v[0] for v in plasma.mesh.vertices ]
        z_vals = [ v[1] for v in plasma.mesh.vertices ]

        # get the squared-euclidean distance using outer subtraction
        D = (subtract.outer(R_vals,R_vals))**2 + (subtract.outer(z_vals,z_vals))**2

        # build covariance matrix using the 'squared-exponential' function
        K = (self.A**2)*exp(-0.5*D/self.L**2)

        # construct the LDL solver for the covariance matrix
        solver = SymmetricSystemSolver(K)

        # use the solver to get the inverse-covariance matrix
        I = eye(K.shape[0])
        self.iK = solver(I)

        # check that the inversion is valid
        error = abs(self.iK.dot(K)-I).max()
        if error > 1e-6:
            raise ValueError('Error in covariance inversion {} exceeds the tolerance of 1e-6'.format(error))

        self.parameters = ['Te', 'ne', 'n0']
        self.parameter_types = ['field']*3

    def __call__(self):
        Te_prior = self.log_prior( log(self.plasma.get('Te')) - self.mu_Te )
        ne_prior = self.log_prior( log(self.plasma.get('ne')) - self.mu_ne )
        n0_prior = self.log_prior( log(self.plasma.get('n0')) - self.mu_n0 )
        return Te_prior + ne_prior + n0_prior

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices['Te']] = self.log_prior_gradient( log(self.plasma.get('Te')) - self.mu_Te ) / self.plasma.get('Te')
        grad[self.plasma.slices['ne']] = self.log_prior_gradient( log(self.plasma.get('ne')) - self.mu_ne ) / self.plasma.get('ne')
        grad[self.plasma.slices['n0']] = self.log_prior_gradient( log(self.plasma.get('n0')) - self.mu_n0 ) / self.plasma.get('n0')
        return grad

    def log_prior(self, field):
        return -0.5*(field.T).dot( self.iK.dot(field) )

    def log_prior_gradient(self, field):
        return -self.iK.dot(field)






class LaplacianSmoothingPrior(object):
    def __init__(self, plasma = None, fields = ('Te','ne','n0'), uncertainty = 0.25):
        self.plasma = plasma
        self.U, distances = plasma.mesh.umbrella_matrix(return_distances = True)

        self.parameters = fields
        self.parameter_types = ['field']*len(self.parameters)

        # normalise to the average edge distances
        distances /= mean(distances)
        self.U = diags(1./distances, format='csc').dot(self.U)
        self.S = (self.U.T).dot(self.U)

        self.lam = 1. / uncertainty**2

    def __call__(self):
        return sum( self.log_prior(self.plasma.get(f)) for f in self.parameters )

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        for f in self.parameters:
            grad[self.plasma.slices[f]] = self.log_prior_gradient(self.plasma.get(f))
        return grad

    def log_prior(self, field):
        df = self.U.dot(log(field))
        return -0.5*self.lam*(df.dot(df))

    def log_prior_gradient(self, field):
        return -self.lam*self.S.dot(log(field)) / field

    def update_uncertainty(self, uncertainty):
        self.lam = 1. / uncertainty**2






class GaussianPrior(object):
    def __init__(self, plasma = None, parameter = None, mean = 0., deviation = 1.):
        self.plasma = plasma

        self.parameter = parameter
        self.parameters = [self.parameter]
        self.mu = mean
        self.sigma = deviation
        self.call_const = -0.5 / deviation**2
        self.grad_const = 2*self.call_const

    def __call__(self):
        f = self.plasma.get(self.parameter)
        return self.call_const * (f - self.mu)**2

    def gradient(self):
        f = self.plasma.get(self.parameter)
        return self.grad_const * (f - self.mu)






class ExponentialFieldPrior(object):
    def __init__(self, plasma=None, field = 'Te', scale=20.):
        self.plasma = plasma
        self.field = field
        self.parameters = [field]
        self.parameter_types = ['field']

        self.lam = 1. / scale

    def __call__(self):
        return self.log_prior(self.plasma.get(self.field))

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices[self.field]] = -self.lam
        return grad

    def log_prior(self, field):
        return -self.lam * npsum(field)

    




class TikhonovPrior(object):
    def __init__(self, plasma=None, fields=('Te', 'ne', 'n0'), operator = None, uncertainty=1.):
        self.plasma = plasma

        self.parameters = fields
        self.parameter_types = ['field'] * len(self.parameters)

        self.U = operator
        self.S = (self.U.T).dot(self.U)

        self.lam = 1. / uncertainty**2

    def __call__(self):
        return sum(self.log_prior(self.plasma.get(f)) for f in self.parameters)

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        for f in self.parameters:
            grad[self.plasma.slices[f]] = self.log_prior_gradient(self.plasma.get(f))
        return grad

    def log_prior(self, field):
        df = self.U.dot(log(field))
        return -0.5 * self.lam * (df.dot(df))

    def log_prior_gradient(self, field):
        return -self.lam * self.S.dot(log(field)) / field

    def update_uncertainty(self, uncertainty):
        self.lam = 1. / uncertainty**2






class TemperatureGradientPrior(object):
    def __init__(self, plasma=None, operator = None, uncertainty=1.):
        self.plasma = plasma

        self.parameters = ['Te']
        self.parameter_types = ['field']

        self.U = operator
        self.lam = 1. / uncertainty**2

    def __call__(self):
        return self.log_prior(self.plasma.get('Te'))

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices['Te']] = self.log_prior_gradient(self.plasma.get('Te'))
        return grad

    def log_prior(self, field):
        df = maximum(self.U.dot(log(field)), 0.)
        return -0.5 * self.lam * (df.dot(df))

    def log_prior_gradient(self, field):
        df = maximum(self.U.dot(log(field)), 0.)
        return -self.lam * (self.U.T).dot(df) / field

    def update_uncertainty(self, uncertainty):
        self.lam = 1. / uncertainty**2






class PressureGradientPrior(object):
    def __init__(self, plasma=None, operator = None, uncertainty=1.):
        self.plasma = plasma

        self.parameters = ['Te', 'ne']
        self.parameter_types = ['field', 'field']

        self.D = operator
        self.lam = 1. / uncertainty**2

    def __call__(self):
        pressure = self.plasma.get('Te') * self.plasma.get('ne')
        return self.log_prior(pressure)

    def gradient(self):
        Te = self.plasma.get('Te')
        ne = self.plasma.get('ne')
        dP = self.log_prior_gradient(ne*Te)
        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices['Te']] = dP*ne
        grad[self.plasma.slices['ne']] = dP*Te
        return grad

    def log_prior(self, field):
        df = maximum(self.D.dot(log(field)), 0.)
        return -0.5 * self.lam * (df.dot(df))

    def log_prior_gradient(self, field):
        df = maximum(self.D.dot(log(field)), 0.)
        return -self.lam * (self.D.T).dot(df) / field

    def update_uncertainty(self, uncertainty):
        self.lam = 1. / uncertainty**2






class PressureDropPrior(object):
    def __init__(self, plasma=None, upstream_indices=None, downstream_indices=None, uncertainty=1.):
        self.plasma = plasma

        self.upstream = upstream_indices
        self.downstream = downstream_indices

        self.parameters = ['Te', 'ne']
        self.parameter_types = ['field', 'field']

        self.lam = 1. / uncertainty**2

    def __call__(self):
        Te = self.plasma.get('Te')
        ne = self.plasma.get('ne')
        P_upstream = Te[self.upstream] * ne[self.upstream]
        P_downstream = Te[self.downstream] * ne[self.downstream]
        z = P_upstream/P_downstream - 2.
        return -0.5*self.lam*(minimum(z,0)**2).sum()

    def gradient(self):
        Te = self.plasma.get('Te')
        ne = self.plasma.get('ne')
        ratio = (Te[self.upstream] * ne[self.upstream]) / (Te[self.downstream] * ne[self.downstream])

        z = ratio - 2.
        base = -self.lam*minimum(z,0) * ratio

        grad = zeros(self.plasma.N_params)
        Te_grad = zeros(self.plasma.n)
        ne_grad = zeros(self.plasma.n)

        Te_grad[self.upstream] = base / Te[self.upstream]
        Te_grad[self.downstream] = -base / Te[self.downstream]
        ne_grad[self.upstream] = base / ne[self.upstream]
        ne_grad[self.downstream] = -base / ne[self.downstream]
        
        grad[self.plasma.slices['Te']] = Te_grad
        grad[self.plasma.slices['ne']] = ne_grad
        return grad

    def update_uncertainty(self, uncertainty):
        self.lam = 1. / uncertainty**2






class CalibrationFactorPrior(object):
    def __init__(self, plasma=None, calibration_tags=None, pixel_weighting=1):
        self.plasma = plasma
        self.parameters = ['calibration_uncertainty']
        self.parameters.extend(calibration_tags)
        self.parameter_types = ['parameter']*len(self.parameters)
        self.pixel_weight = pixel_weighting

    def __call__(self):
        sigma = self.plasma.get('calibration_uncertainty')[0]
        cals = array([self.plasma.get(tag)[0] for tag in self.parameters[1:]])
        z = (cals-1.)/sigma
        return (-0.5*(z**2).sum() - cals.size*log(sigma))*self.pixel_weight

    def gradient(self):
        sigma = self.plasma.get('calibration_uncertainty')[0]
        cals = array([self.plasma.get(tag)[0] for tag in self.parameters[1:]])
        z = (cals-1.)/sigma
        dLdc = -(self.pixel_weight/sigma)*z

        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices['calibration_uncertainty']] = (self.pixel_weight/sigma)*(z**2 - 1.).sum()
        for tag, deriv in zip(self.parameters[1:], dLdc):
            grad[self.plasma.slices[tag]] = deriv
        return grad






class QGradientPrior(object):
    def __init__(self, plasma=None, operator=None, uncertainty=1.):
        self.plasma = plasma

        self.parameters = ['Te']
        self.parameter_types = ['field']

        self.U = operator
        self.lam = 1. / uncertainty**2

    def __call__(self):
        return self.log_prior(self.plasma.get('Te')**3.5)

    def gradient(self):
        grad = zeros(self.plasma.N_params)
        Te = self.plasma.get('Te')
        grad[self.plasma.slices['Te']] = 3.5*(Te**2.5)*self.log_prior_gradient(Te**3.5)
        return grad

    def log_prior(self, field):
        df = maximum(self.U.dot(field), 0.)
        return -0.5 * self.lam * (df.dot(df))

    def log_prior_gradient(self, field):
        df = maximum(self.U.dot(field), 0.)
        return -self.lam * (self.U.T).dot(df)

    def update_uncertainty(self, uncertainty):
        self.lam = 1. / uncertainty**2