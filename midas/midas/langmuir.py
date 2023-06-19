
from numpy import array, load, zeros, diag, sqrt, linspace
from scipy.sparse import csc_matrix


class LangmuirLikelihood(object):
    """
    langmuir posterior
    """
    def __init__(self, plasma = None, data_dir = None):
        self.plasma = plasma

        self.parameters = ['Te', 'ne']
        self.parameter_types = ['field', 'field']

        if data_dir is not None:
            data = load(data_dir)
            self.Te = data['Te']
            self.Te_err = data['Te_err']
            self.ne = data['ne']
            self.ne_err = data['ne_err']

            R = data['R']
            z = data['z']
            self.data_coords = list(zip(R,z))
            self.G = plasma.mesh.build_interpolator_matrix(self.data_coords)

            self.W_T = diag(self.Te_err**(-2))
            self.W_n = diag(self.ne_err**(-2))

            self.H_T = (self.G.T).dot( self.W_T.dot( self.G ) )
            self.H_n = (self.G.T).dot( self.W_n.dot( self.G ) )

            self.D_T = (self.G.T).dot(self.W_T.dot(self.Te))
            self.D_n = (self.G.T).dot(self.W_n.dot(self.ne))

    def __call__(self):
        fwd_T = self.G.dot(self.plasma.get('Te'))
        fwd_n = self.G.dot(self.plasma.get('ne'))

        temp_LP = -0.5*((fwd_T - self.Te) / self.Te_err)**2
        dens_LP = -0.5*((fwd_n - self.ne) / self.ne_err)**2

        return temp_LP.sum() + dens_LP.sum()

    def forward(self):
        fwd_T = self.G.dot(self.plasma.get('Te'))
        fwd_n = self.G.dot(self.plasma.get('ne'))
        return fwd_T, fwd_n

    def gradient(self):
        grad_T = -self.H_T.dot(self.plasma.get('Te')) + self.D_T
        grad_n = -self.H_n.dot(self.plasma.get('ne')) + self.D_n

        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices['Te']] = grad_T
        grad[self.plasma.slices['ne']] = grad_n

        return grad






class JsatLikelihood(object):
    """
    langmuir posterior
    """
    def __init__(self, plasma = None, data_dir = None, sparse_operators = True):
        self.plasma = plasma

        self.parameters = ['Te', 'ne']
        self.parameter_types = ['field', 'field']

        # build the proportionality constant for the J-sat model
        e = 1.602e-19
        mi = 2 * 1.6605e-27
        self.alpha = 0.5*sqrt(2*e/mi)

        if data_dir is not None:
            data = load(data_dir)
            self.Jsat = data['jsat']
            self.Jsat_err = data['jsat_err']
            self.ivar = 1./self.Jsat_err**2

            R = data['R']
            z = data['z']
            self.data_coords = list(zip(R,z))
            self.G = plasma.mesh.build_interpolator_matrix(self.data_coords)
            if sparse_operators:
                self.G = csc_matrix(self.G)

    def __call__(self):
        J_pred = self.forward()
        LP = ((self.Jsat - J_pred) / self.Jsat_err)**2
        return -0.5*LP.sum()

    def jsat_model(self, Te, ne):
        return self.alpha * ne * sqrt(Te)

    def forward(self):
        fwd_T = self.G.dot(self.plasma.get('Te'))
        fwd_n = self.G.dot(self.plasma.get('ne'))
        return self.jsat_model(fwd_T, fwd_n)

    def gradient(self):
        fwd_T = self.G.dot(self.plasma.get('Te'))
        fwd_n = self.G.dot(self.plasma.get('ne'))
        J = self.jsat_model(fwd_T, fwd_n)

        beta = self.ivar*(self.Jsat - J)*self.alpha

        dL_dT = self.G.T.dot( 0.5*beta*fwd_n/sqrt(fwd_T) )
        dL_dn = self.G.T.dot( beta*fwd_T )

        grad = zeros(self.plasma.N_params)
        grad[self.plasma.slices['Te']] = dL_dT
        grad[self.plasma.slices['ne']] = dL_dn
        return grad






def mastu_t5_langmuir_coords():

    x0 = array((1.350, -2.060)) # point at the join of tile 4 and 5
    x1 = array((1.654, -1.756)) # point further up tile 5

    # unit vectors pointing along tiles 4 & 5
    unit_4 = array([-1, 0])
    unit_5 = (x1-x0) / sqrt(sum((x1-x0)**2))

    t5_lengths = linspace(1,36,36) * 0.01
    t5_points = [x0 + L*unit_5 for L in t5_lengths]

    p = []
    p.extend(t5_points)

    for v in p:
        v[1] += 0.001
        v[0] -= 0.001

    return p



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    p = mastu_t5_langmuir_coords()
    plt.plot([i[0] for i in p], [i[0] for i in p], '.')
    plt.grid()
    plt.show()