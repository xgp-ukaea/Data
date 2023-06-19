
from numpy import load, zeros, diag


class ThomsonLikelihood(object):

    def __init__(self, plasma = None, data_dir = None):
        self.plasma = plasma

        self.parameters = ['Te', 'ne']
        self.parameter_types = ['field', 'field']

        # settings dictionary specifies data source
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

            self.W_T = diag(self.Te_err ** (-2))
            self.W_n = diag(self.ne_err ** (-2))

            self.H_T = (self.G.T).dot( self.W_T.dot( self.G ) )
            self.H_n = (self.G.T).dot( self.W_n.dot( self.G ) )

            self.D_T = (self.G.T).dot(self.W_T.dot(self.Te))
            self.D_n = (self.G.T).dot(self.W_n.dot(self.ne))

    def __call__(self):
        fwd_T = self.G.dot(self.plasma.get('Te'))
        fwd_n = self.G.dot(self.plasma.get('ne'))

        temp_LP = -0.5* ((fwd_T - self.Te) / self.Te_err) ** 2
        dens_LP = -0.5* ((fwd_n - self.ne) / self.ne_err) ** 2
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


mastu_ts_points = [
    (1.00049, -1.80450), (1.01004, -1.82050), (1.03148, -1.83650),
    (1.06409, -1.85250), (1.10689, -1.86850), (1.15875, -1.88450),
    (1.21852, -1.90050), (1.28508, -1.91650), (1.35744, -1.93250),
    (1.43473, -1.94850) ]