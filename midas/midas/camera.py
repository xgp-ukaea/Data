
from numpy import array, load, zeros, exp, maximum, sqrt, log, dot
from scipy.sparse import diags
from midas.model_data import line_parameter_lookup
from midas.emission import construct_emission_model


class EmissivityFieldLikelihood(object):
    """
    Posterior class for MSI filtered cameras
    """
    def __init__(self, plasma=None, line=None, data_dir=None, free_calibration=False, calibration_tag=None):

        self.plasma = plasma
        self.filter = line

        # settings file would specify data source for each diagnostic posterior
        if data_dir is not None:
            data = load(data_dir)
            self.emissivity = data['emissivity']
            self.error = data['error']
            self.inv_sigma = 1. / self.error
            self.inv_sigma_sq = self.inv_sigma**2

        self.spline = construct_emission_model(line)

        self.emission_parameters = line_parameter_lookup[line]
        self.emission_parameter_types = ['field'] * len(self.emission_parameters)

        if free_calibration:
            if calibration_tag is None:
                self.calibration_tag = line + '_camera_calibration'
            else:
                self.calibration_tag = calibration_tag

            self.parameters = [*self.emission_parameters, self.calibration_tag]
            self.parameter_types = [*self.emission_parameter_types, 'parameter']
            self.call_alias = self.calibrated_call
            self.gradient = self.calibrated_gradient
        else:
            self.parameters = self.emission_parameters
            self.parameter_types = self.emission_parameter_types
            self.call_alias = self.uncalibrated_call
            self.gradient = self.uncalibrated_gradient

    def __call__(self):
        return self.call_alias()

    def uncalibrated_call(self):
        prediction = self.forward()
        return -0.5*(((prediction - self.emissivity)*self.inv_sigma)**2).sum()

    def calibrated_call(self):
        prediction = self.forward()
        cal = self.plasma.get(self.calibration_tag)
        return -0.5*(((prediction*cal - self.emissivity)*self.inv_sigma)**2).sum()

    def uncalibrated_gradient(self):
        # get the emissivity prediction and its partial derivatives w.r.t. the parameters
        params = [self.plasma.get(p) for p in self.emission_parameters]
        prediction, partials = self.spline.emission_and_gradient(*params)
        # calculate the derivative of the log-probability w.r.t. the emissivity
        dL_dE = (self.emissivity - prediction)*self.inv_sigma_sq
        # calculate the gradient of the log-probability via the chain rule
        grad = zeros(self.plasma.N_params)

        for param in self.parameters:
            grad[self.plasma.slices[param]] = partials[param] * dL_dE
        return grad

    def calibrated_gradient(self):
        # get the emissivity prediction and its partial derivatives w.r.t. the parameters
        params = [self.plasma.get(p) for p in self.emission_parameters]
        cal = self.plasma.get(self.calibration_tag)
        prediction, partials = self.spline.emission_and_gradient(*params)
        # calculate the derivative of the log-probability w.r.t. the emissivity

        A = (self.emissivity - cal * prediction) * self.inv_sigma_sq
        dL_dE = cal*A
        dL_dc = A.dot(prediction)
        # calculate the gradient of the log-probability via the chain rule
        grad = zeros(self.plasma.N_params)

        for param in self.emission_parameters:
            grad[self.plasma.slices[param]] = partials[param] * dL_dE
        grad[self.plasma.slices[self.calibration_tag]] = dL_dc
        return grad

    def forward(self):
        params = [self.plasma.get(p) for p in self.emission_parameters]
        emissivity = self.spline(*params)
        return emissivity






class UncalibratedCameraLikelihood(object):
    def __init__(self, plasma=None, line=None, data_dir=None, geometry_matrix=None, error_coeffs=None, weighting = 1., calibration = 1.):
        self.plasma = plasma
        self.G = geometry_matrix
        self.weight = weighting

        self.line = line
        self.spline = construct_emission_model(line)

        self.parameters = line_parameter_lookup[line]
        self.parameter_types = ['field']*len(self.parameters)

        if data_dir is not None:
            data = load(data_dir)
            self.brightness = data['brightness']*calibration
            self.image_shape = data['image_shape']
            if error_coeffs is None:
                self.error = data['error']*calibration
            else:
                fmax, f0 = error_coeffs
                bmax = self.brightness.max()
                self.error = sqrt(self.brightness*bmax*(fmax**2 - f0**2) + (bmax*f0)**2)

            self.inv_sigma = 1./self.error

            iS = diags(self.inv_sigma**2, 0, format=None)

            Q = (self.G.T).dot(iS)
            self.A = -Q.dot(self.G)
            self.B = Q.dot(self.brightness)

    def forward(self):
        params = [self.plasma.get(p) for p in self.parameters]
        emissivity = self.spline(*params)
        return self.G.dot(emissivity)

    def gradient(self):
        params = [self.plasma.get(p) for p in self.parameters]
        emissivity, partials = self.spline.emission_and_gradient(*params)

        dL_dE = self.A.dot(emissivity) + self.B
        grad = zeros(self.plasma.N_params)

        for param in self.parameters:
            grad[self.plasma.slices[param]] = partials[param] * dL_dE
        return self.weight*grad

    def __call__(self):
        prediction = self.forward()
        return -0.5*self.weight*(((prediction - self.brightness) * self.inv_sigma)**2).sum()






class FilteredCameraLikelihood(object):
    def __init__(self, plasma=None, line=None, data_dir=None, geometry_matrix=None, error_coeffs = None, calibration_tag = None):
        self.plasma = plasma
        self.G = geometry_matrix

        self.line = line
        self.spline = construct_emission_model(line)

        if calibration_tag is None:
            self.calibration_tag = line + '_camera_calibration'
        else:
            self.calibration_tag = calibration_tag

        self.emission_parameters = line_parameter_lookup[line]
        self.emission_parameter_types = ['field']*len(self.emission_parameters)

        self.parameters = [*self.emission_parameters, self.calibration_tag]
        self.parameter_types = [*self.emission_parameter_types, 'parameter']

        if data_dir is not None:
            data = load(data_dir)
            self.brightness = data['brightness']

            if error_coeffs is None:
                self.error = data['error']
            else:
                fmax, f0 = error_coeffs
                bmax = self.brightness.max()
                self.error = sqrt(self.brightness*bmax*(fmax**2 - f0**2) + (bmax*f0)**2)

            self.inv_sigma = 1./self.error

            iS = diags(self.inv_sigma**2, 0, format=None)

            Q = (self.G.T).dot(iS)
            self.A = -Q.dot(self.G)
            self.B = Q.dot(self.brightness)
            self.d = -self.brightness.T.dot( iS.dot(self.brightness) )

    def forward(self):
        params = [self.plasma.get(p) for p in self.emission_parameters]
        emissivity = self.spline(*params)
        return self.G.dot(emissivity)

    def gradient(self):
        params = [self.plasma.get(p) for p in self.emission_parameters]
        f, partials = self.spline.emission_and_gradient(*params)
        c = self.plasma.get(self.calibration_tag)

        dL_df = self.A.dot(f) + c*self.B
        dL_dc = self.B.T.dot(f) + c*self.d

        grad = zeros(self.plasma.N_params)
        for param in self.emission_parameters:
            grad[self.plasma.slices[param]] = partials[param] * dL_df
        grad[self.plasma.slices[self.calibration_tag]] = dL_dc
        return grad

    def __call__(self):
        prediction = self.forward()
        cali = self.plasma.get(self.calibration_tag)
        return -0.5*(((prediction - cali*self.brightness) * self.inv_sigma)**2).sum()






class CorrelatedCameraLikelihood(object):
    def __init__(self, plasma=None, line=None, data_dir=None, geometry_matrix=None, error_coeffs=None, systematic_error=0.):
        self.plasma = plasma
        self.G = geometry_matrix

        self.line = line
        self.spline = construct_emission_model(line)

        self.parameters = line_parameter_lookup[line]
        self.parameter_types = ['field']*len(self.parameters)

        if data_dir is not None:
            data = load(data_dir)
            self.brightness = data['brightness']
            self.image_shape = data['image_shape']
            if error_coeffs is None:
                self.error = data['error']
            else:
                fmax, f0 = error_coeffs
                bmax = self.brightness.max()
                self.error = sqrt(self.brightness*bmax*(fmax**2 - f0**2) + (bmax*f0)**2)

            self.inv_sigma = 1./self.error

            iS = diags(self.inv_sigma**2, 0, format=None)

            # uncorrelated error pre-calcs
            Q = (self.G.T).dot(iS)
            self.A = -Q.dot(self.G)
            self.B = Q.dot(self.brightness)

            # correlated error pre-calcs
            self.sys = self.brightness*systematic_error
            self.r = sqrt(1 + (self.sys**2 / self.error**2).sum())
            self.q = self.sys / (self.r * self.error**2)
            self.u = (self.G.T).dot(self.q)
            self.h = dot(self.q, self.brightness)

            # normalisation
            self.norm = -log(self.r) - log(self.error).sum()

    def forward(self):
        emissivity = self.spline(self.plasma.get('Te'),
                                 self.plasma.get('ne'),
                                 self.plasma.get(self.neutral_tag))
        return self.G.dot(emissivity)

    def gradient(self):
        emissivity, dE_dTe, dE_dne, dE_dn0 = self.spline.emission_and_gradient(
            self.plasma.get('Te'),
            self.plasma.get('ne'),
            self.plasma.get(self.neutral_tag)
        )

        C = self.A.dot(emissivity) + self.B + (dot(self.u,emissivity) - self.h)*self.u
        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices['Te']] = dE_dTe * C
        grad[self.plasma.slices['ne']] = dE_dne * C
        grad[self.plasma.slices[self.neutral_tag]] = dE_dn0 * C
        return grad

    def __call__(self):
        dz = self.forward() - self.brightness
        return -0.5*((dz*self.inv_sigma)**2).sum() + 0.5*dot(self.q,dz)**2 + self.norm






class EmissivityPosterior(object):
    def __init__(self, emissivities = None, errors = None, lines = None, bounds = None):

        self.emissivities = emissivities
        self.errors = errors
        if bounds is None:
            self.__call__ = self.unbounded_call
        else:
            self.lower = array([k[0] for k in bounds])
            self.upper = array([k[1] for k in bounds])
            self.__call__ = self.bounded_call

        if lines is None:
            self.lines = ['D_alpha', 'D_beta', 'D_gamma', 'D_delta']
        else:
            self.lines = lines

        self.models = []
        for line in self.lines:
            self.models.append( construct_emission_model(line) )

        self.parameters = []
        for model in self.models:
            for p in model.emission_parameters:
                if p not in self.parameters:
                    self.parameters.append(p)

        self.parameter_map = {p:i for i,p in enumerate(self.parameters)}

        # uncertainty model
        self.error_model = self.gaussian_error

        # specify components of the posterior
        self.posterior_components = [ self.likelihood,
                                      self.max_pressure_prior,
                                      self.density_ratio_prior ]
                                      # self.neutral_fraction_prior ]

        # max pressure constraint settings
        self.pe_max = 7e20
        self.pe_std = self.pe_max*0.05

        # neutral fraction prior settings
        self.frac_lambda = 0.2
        self.frac_sigma = 0.1
        self.frac_b = 0.035

        # density ratio prior settings
        self.ratio_lambda = 0.3
        self.ratio_sigma = 0.1
        self.ratio_a = 4.
        self.ratio_b = 0.05

    def max_pressure_prior(self, theta):
        Te, ne = [theta[self.parameter_map[key]] for key in ('Te', 'ne') ]
        dp = maximum(Te*ne - self.pe_max, 0.)
        return -0.5*(dp/self.pe_std)**2

    def density_ratio_prior(self, theta):
        Te, ne, n0 = [theta[self.parameter_map[key]] for key in ('Te', 'ne', 'H0') ]
        r = n0 / ne
        r_max = self.ratio_a * exp(-self.ratio_lambda*Te) + self.ratio_b
        dp = maximum((r / r_max) - 1, 0.)
        return -0.5*(dp/self.ratio_sigma)**2

    def neutral_fraction_prior(self, theta):
        Te, ne, n0 = [theta[self.parameter_map[key]] for key in ('Te', 'ne', 'H0') ]
        f = n0 / (n0 + ne)
        f_max = (1-self.frac_b)*exp(-self.frac_lambda*Te) + self.frac_b
        dp = maximum(f/f_max - 1, 0.)
        return -0.5*(dp / self.frac_sigma)**2

    def reciprocal_prior(self, theta):
        return -log(theta).sum()

    def gaussian_error(self, prediction, data, error):
        Z = (data - prediction)/error
        return -0.5*Z**2

    def likelihood(self, theta):
        predictions = [ model(*[theta[self.parameter_map[p]] for p in model.emission_parameters]) for model in self.models ]
        return sum( self.error_model(*v) for v in zip(predictions, self.emissivities, self.errors) )

    def bounded_call(self, theta):
        if ((self.lower < theta) & (self.upper > theta)).all():
            return sum( p(theta) for p in self.posterior_components )
        else:
            return -1e10

    def unbounded_call(self, theta):
        return sum( p(theta) for p in self.posterior_components )

    def log_theta_call(self, log_theta):
        theta = exp(log_theta)
        return sum(p(theta) for p in self.posterior_components)

    def log_theta_bounded_call(self, log_theta):
        theta = exp(log_theta)
        if ((self.lower < theta) & (self.upper > theta)).all():
            return sum( p(theta) for p in self.posterior_components )
        else:
            return -1e10

    def cost(self, theta):
        return -self.__call__(theta)

