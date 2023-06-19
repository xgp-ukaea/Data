
from numpy import sqrt, sin, cos, arccos, pi
from numpy import array, linspace, load, diag, zeros
from scipy.sparse import diags as diag_sparse

from mesh_tools.geometry import GeometryMatrix
from midas.model_data import line_parameter_lookup
from midas.emission import construct_emission_model


def spectrometer_chords():
    # system 1 properties
    N_chords = 20
    origin = (2.13925, 0., -1.320)
    end = (1.0347, 0., -2.006)
    fov = (17.3/360)*2*pi
    origins = []

    # work out central angle
    central_dir = array(end)-array(origin)
    central_dir /= sqrt(central_dir.dot(central_dir))
    phi0 = -arccos(central_dir[0])

    # generate the chords
    phi = linspace(phi0-0.5*fov, phi0+0.5*fov, N_chords)
    directions = [array([cos(p), 1e-6, sin(p)]) for p in phi]
    origins.extend( [array(origin)]*N_chords )

    # now repeat process for system 2
    N_chords = 20
    origin = (2.14225, 0., -1.5)
    end = (0.843745, 0., -1.815)
    fov = (15.2 / 360)*2*pi

    # work out central angle
    central_dir = array(end) - array(origin)
    central_dir /= sqrt(central_dir.dot(central_dir))
    phi0 = -arccos(central_dir[0])

    # generate the chords
    phi = linspace(phi0 - 0.5*fov, phi0 + 0.5*fov, N_chords)
    directions.extend( [array([cos(p), 1e-6, sin(p)]) for p in phi] )
    origins.extend( [array(origin)] * N_chords )

    directions = [v / sqrt(v.dot(v)) for v in directions]

    return origins, directions




def build_chords(R,z,t):
    directions = [array([-cos(x), 1e-6, sin(x)]) for x in t]
    origins = [array([a, 0., b]) for a,b in zip(R,z)]
    lengths = [1.5]*len(R)
    return origins, directions, lengths




class IntegratedSpectroscopyLikelihood(object):
    def __init__(self, plasma = None, line = None, data_dir = None, sparse_operators = True):
        self.plasma = plasma

        self.neutral_tag = line_parameter_lookup[line]
        self.spline = construct_emission_model(line)

        self.parameters = line_parameter_lookup[line]
        self.parameter_types = ['field']*len(self.parameters)

        if data_dir is not None:
            # extract data
            data = load(data_dir)
            self.brightness = data['brightness']
            self.density = data['density']
            self.inv_b_error = 1./data['brightness_error']
            self.inv_d_error = 1./data['density_error']
            self.invsqr_d_error = self.inv_d_error**2

            # build the geometry matrix
            self.build_geometry_matrix(data['ray_origins'], data['ray_directions'], data['ray_lengths'])

            if sparse_operators:
                iS = diag_sparse(self.inv_b_error**2)
            else:
                iS = diag(self.inv_b_error**2)
                self.G = self.G.toarray()

            Q = (self.G.T).dot(iS)
            self.A_b = -Q.dot(self.G)
            self.B_b = Q.dot(self.brightness)

    def build_geometry_matrix(self, origins, directions, lengths):
        Gcalc = GeometryMatrix(trimesh = self.plasma.mesh, ray_origins = origins,
                               ray_directions = directions, ray_lengths = lengths)
        self.G = Gcalc.run()

    def forward(self):
        params = [self.plasma.get(p) for p in self.parameters]
        emissivity = self.spline(*params)
        b = self.G.dot(emissivity)
        return b, self.G.dot(emissivity*self.plasma.get('ne')) / b

    def gradient(self):
        # get the emissivity and its gradients
        params = [self.plasma.get(p) for p in self.parameters]
        emissivity, partials = self.spline.emission_and_gradient(*params)
        ne = self.plasma.get('ne')
        # get the coefficient for the brightness
        b_coeff = self.A_b.dot(emissivity) + self.B_b

        # now work out weighted density coefficient
        b = self.G.dot(emissivity)
        w = self.G.dot(emissivity*ne) / b
        z = (self.density - w)*self.invsqr_d_error/b

        p1 = self.G.T.dot(z)
        p2 = self.G.T.dot(z*w)
        d_coeff = ne*p1 - p2

        # combine into a total coefficient
        total_coeff = b_coeff + d_coeff

        # fill the gradient vector
        grad = zeros(self.plasma.N_params)
        for param in self.parameters:
            grad[self.plasma.slices[param]] = partials[param] * total_coeff

        # additional term to deal with ne dependence of density estimate
        grad[self.plasma.slices['ne']] += p1*emissivity

        return grad

    def __call__(self):
        brightness, density = self.forward()
        L_brightness = -0.5*(((brightness - self.brightness)*self.inv_b_error)**2).sum()
        L_density = -0.5*(((density - self.density)*self.inv_d_error)**2).sum()
        return L_brightness + L_density



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mesh_tools.vessel_boundaries import superx_boundary
    from mesh_tools.geometry import PixelRay

    plt.plot(*superx_boundary(), lw = 2, color = 'black')

    origins, directions = spectrometer_chords()
    lengths = [1.6]*len(origins)

    L = linspace(0.3, 1.6, 2000)
    L_integral = []

    raygen = (PixelRay(start = S, direction = D, length = L) for S, D, L in zip(origins, directions, lengths))
    for ray in raygen:
        R, z = ray(L)
        plt.plot(R,z)

    plt.axis('equal')
    plt.xlabel('R (m)')
    plt.ylabel('z (m)')
    plt.tight_layout()
    plt.show()