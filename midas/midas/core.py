
from numpy import isfinite


class MidasPosterior(object):
    """
    docstring for IdaPosterior
    """
    def __init__(self, plasma = None, diagnostics = None, priors = None):
        # store the state object
        self.plasma = plasma
        # gather the posterior components together
        self.diagnostics = diagnostics
        self.diagnostics.extend(priors)

        # inspect all the diagnostics and priors to build the parameter list
        self.plasma.build_parametrisation(self.diagnostics)

    def __call__(self, theta):
        self.plasma.update_state(theta) # update the plasma state with proposed parameters
        return sum( component() for component in self.diagnostics )

    def gradient(self, theta):
        self.plasma.update_state(theta) # update the plasma state with proposed parameters
        return sum( component.gradient() for component in self.diagnostics )

    def get_parameters(self):
        D = dict()
        for tag in self.plasma.parameters.keys():
            D[tag] = self.plasma.get(tag)
        return D

    def fitness(self, theta):
        lpl = self.__call__(theta)
        if isfinite(lpl):
            return [-lpl]
        else:
            return[1e50]








