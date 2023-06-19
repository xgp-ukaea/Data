

class PlasmaState(object):
    """
    this 'Plasma' object could serve as the interface to the plasma solution,
    and would contain the TriangularMesh object or some other representation
    of the plasma solution
    """
    def __init__(self, mesh = None):
        # store the mesh
        self.mesh = mesh
        self.n = len(self.mesh.vertices)

        # storage for parameter vector
        self.theta = None

    def build_parametrisation(self, components):
        # inspect all the posterior components and extract all fields
        # and parameters that are required
        self.parameters = {}
        for c in components:
            for tag, typ in zip( c.parameters, c.parameter_types ):
                if tag not in self.parameters:
                    self.parameters[tag] = typ
                elif self.parameters[tag] != typ:
                    raise ValueError(
                        """
                        Two matching parameter tags have different variable types:
                        >> tag '{}' has been assigned both '{}' and '{}' as types.
                        """.format(tag, typ, self.parameters[tag]))

        # sort by type, then by variable name
        pairs = sorted(self.parameters.items(), key = lambda x : x[0])
        pairs = sorted(pairs, key = lambda x : x[1])
        # replace the types with the variable sizes
        type_lengths = {'field': self.n, 'parameter': 1}
        pairs = [ (par, type_lengths[typ]) for par, typ in pairs ]
        # now build pairs of parameter names and slice objects
        slices = []
        for p, L in pairs:
            if len(slices) is 0:
                slices.append( (p, slice(0,L)) )
            else:
                last = slices[-1][1].stop
                slices.append( (p, slice(last, last+L)) )

        # the stop field of the last slice is the total number of parameter values
        self.N_params = slices[-1][1].stop
        # convert to a dictionary which maps parameter names to corresponding
        # slices of the parameter vector
        self.slices = dict(slices)

    def update_state(self, theta):
        self.theta = theta

    def get(self, tag):
        return self.theta[self.slices[tag]]