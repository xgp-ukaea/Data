from numpy import load, log, exp, meshgrid
from scipy.interpolate import RectBivariateSpline as spl2d
from midas.model_data import atomic_data_lookup, molecular_data_lookup, line_parameter_lookup
from midas.model_data import line_parameter_molecular_lookup, hydrogen_model_lines, impurity_model_lines


def construct_emission_model(line, is_include_mol_effects=False):
    """
    Parameters:
        :param (str) line: the line for which the emission model is desired.
            See 'model_data/__init__.py' for options
        :param (bool) is_include_mol_effects: whether molecular emission is desired to be included
            in the model or not
    Returns:
        :return: a model for the specified line based on ADAS PEC data
    """
    if line in hydrogen_model_lines and is_include_mol_effects:
        return AdasHydrogenMolecularModel.build(line)
    elif line in hydrogen_model_lines and not is_include_mol_effects:
        return AdasHydrogenModel.build(line)
    elif line in impurity_model_lines and not is_include_mol_effects:
        return AdasImpurityModel.build(line)
    else:
        msg = '{} ({} molecular effects) is not a valid atomic line identifier'.format(
            line, 'with' if is_include_mol_effects else 'without')
        raise KeyError(msg)


class AdasHydrogenModel(object):
    def __init__(self, ln_ne=None, ln_te=None, rec_ln_pec=None, exc_ln_pec=None, emission_parameters=None):

        log_ne_mesh, __ = meshgrid(ln_ne, ln_te, indexing='ij')
        self.emission_parameters = emission_parameters

        # build the splines
        self.log_excite = spl2d(ln_ne, ln_te, exc_ln_pec + log_ne_mesh)
        self.log_recomb = spl2d(ln_ne, ln_te, rec_ln_pec + 2*log_ne_mesh)

        # store the limit of the spline data
        self.ne_lims = [exp(min(ln_ne)), exp(max(ln_ne))]
        self.te_lims = [exp(min(ln_te)), exp(max(ln_te))]

    def __call__(self, te, ne, n0):
        log_te = log(te)
        log_ne = log(ne)
        E_rec = exp( self.log_recomb.ev(log_ne, log_te) )
        E_exc = exp( self.log_excite.ev(log_ne, log_te) ) * n0
        return E_rec + E_exc

    # def vectorised(self, te, ne, n0):
    #     ln_te = log(te)
    #     ln_ne = log(ne)
    #     E_rec = exp( self.log_recomb.ev(ln_ne, ln_te) )
    #     E_exc = exp( self.log_excite.ev(ln_ne, ln_te) ) * n0
    #     return E_rec + E_exc

    def recombination(self, te, ne, n0):
        ln_te = log(te)
        ln_ne = log(ne)
        return exp( self.log_recomb.ev(ln_ne, ln_te) )

    def excitation(self, te, ne, n0):
        ln_te = log(te)
        ln_ne = log(ne)
        return exp( self.log_excite.ev(ln_ne, ln_te) ) * n0

    # def vec_logged(self, ln_te, ln_ne, n0):
    #     E_rec = exp( self.log_recomb.ev(ln_ne, ln_te) )
    #     E_exc = exp( self.log_excite.ev(ln_ne, ln_te) ) * n0
    #     return E_rec + E_exc

    def log_model(self, ln_te, ln_ne, ln_n0):
        E_rec = exp( self.log_recomb.ev(ln_ne, ln_te) )
        E_exc = exp( self.log_excite.ev(ln_ne, ln_te) + ln_n0)
        return E_rec + E_exc

    @classmethod
    def load(cls,path):
        data = load(path)
        model = cls(ln_ne = data['ln_ne'],
                    ln_te = data['ln_te'],
                    exc_ln_pec = data['exc_ln_pec'],
                    rec_ln_pec = data['rec_ln_pec'])
        return model

    @classmethod
    def build(cls,line):
        if line not in atomic_data_lookup:
            raise KeyError('{} is not a valid atomic line identifier'.format(line))

        data = load(atomic_data_lookup[line])
        model = cls(ln_ne = data['ln_ne'],
                    ln_te = data['ln_te'],
                    exc_ln_pec = data['exc_ln_pec'],
                    rec_ln_pec = data['rec_ln_pec'],
                    emission_parameters = line_parameter_lookup[line])
        return model

    def gradient(self, te, ne, n0):
        ln_te = log(te)
        ln_ne = log(ne)

        rec = exp( self.log_recomb.ev(ln_ne, ln_te) )
        dE_dn0 = exp( self.log_excite.ev(ln_ne, ln_te) )

        df_dn = self.log_recomb.ev(ln_ne, ln_te, dx=1)
        df_dT = self.log_recomb.ev(ln_ne, ln_te, dy=1)

        dh_dn = self.log_excite.ev(ln_ne, ln_te, dx=1)
        dh_dT = self.log_excite.ev(ln_ne, ln_te, dy=1)

        exc = n0*dE_dn0
        dE_dn = (rec*df_dn + exc*dh_dn)/ne
        dE_dT = (rec*df_dT + exc*dh_dT)/te
        # Emissivity gradient wrt temperature, density and emitter neutral density
        return {'Te' : dE_dT, 'ne' : dE_dn, 'n0' : dE_dn0}

    def emission_and_gradient(self, te, ne, n0):
        ln_te = log(te)
        ln_ne = log(ne)

        R = exp( self.log_recomb.ev(ln_ne, ln_te) )
        dE_dn0 = exp( self.log_excite.ev(ln_ne, ln_te) )
        X = dE_dn0*n0

        dR_dn = self.log_recomb.ev(ln_ne, ln_te, dx=1)
        dR_dT = self.log_recomb.ev(ln_ne, ln_te, dy=1)

        dX_dn = self.log_excite.ev(ln_ne, ln_te, dx=1)
        dX_dT = self.log_excite.ev(ln_ne, ln_te, dy=1)

        dE_dn = (R*dR_dn + X*dX_dn)/ne
        dE_dT = (R*dR_dT + X*dX_dT)/te
        # Emissivity, and gradient wrt temperature, density and emitter neutral density
        return R+X, {'Te' : dE_dT, 'ne' : dE_dn, 'n0' : dE_dn0}


class AdasImpurityModel(object):
    def __init__(self, ln_ne = None, ln_te = None, rec_ln_pec = None, exc_ln_pec = None, emission_parameters = None):

        log_ne_mesh, __ = meshgrid(ln_ne, ln_te, indexing='ij')
        self.emission_parameters = emission_parameters

        # build the splines
        self.log_excite_pec = spl2d(ln_ne, ln_te, exc_ln_pec + log_ne_mesh)
        self.log_recomb_pec = spl2d(ln_ne, ln_te, rec_ln_pec + log_ne_mesh)

        # store the limit of the spline data
        self.ne_lims = [exp(min(ln_ne)), exp(max(ln_ne))]
        self.te_lims = [exp(min(ln_te)), exp(max(ln_te))]

    def __call__(self, te, ne, n_lower, n_upper):
        ln_te = log(te)
        ln_ne = log(ne)
        E_rec = exp(self.log_recomb_pec.ev(ln_ne, ln_te)) * n_upper
        E_exc = exp(self.log_excite_pec.ev(ln_ne, ln_te)) * n_lower
        return E_rec + E_exc

    # def vectorised(self, te, ne, n_lower, n_upper):
    #     ln_te = log(te)
    #     ln_ne = log(ne)
    #     E_rec = exp( self.log_recomb.ev(ln_ne, ln_te) )
    #     E_exc = exp( self.log_excite.ev(ln_ne, ln_te) ) * n0
    #     return E_rec + E_exc

    def recombination(self, te, ne, n_lower, n_upper):
        ln_te = log(te)
        ln_ne = log(ne)
        return exp(self.log_recomb_pec.ev(ln_ne, ln_te)) * n_upper

    def excitation(self, te, ne, n_lower, n_upper):
        ln_te = log(te)
        ln_ne = log(ne)
        return exp(self.log_excite_pec.ev(ln_ne, ln_te)) * n_lower

    # def vec_logged(self, ln_te, ln_ne, n0):
    #     E_rec = exp( self.log_recomb.ev(ln_ne, ln_te) )
    #     E_exc = exp( self.log_excite.ev(ln_ne, ln_te) ) * n0
    #     return E_rec + E_exc

    def log_model(self, ln_te, ln_ne, ln_n_lower, ln_n_upper):
        E_rec = exp(self.log_recomb_pec.ev(ln_ne, ln_te) + ln_n_upper)
        E_exc = exp(self.log_excite_pec.ev(ln_ne, ln_te) + ln_n_lower)
        return E_rec + E_exc

    # @classmethod
    # def load(cls,path):
    #     data = load(path)
    #     model = cls(ln_ne = data['ln_ne'],
    #                 ln_te = data['ln_te'],
    #                 exc_ln_pec = data['exc_ln_pec'],
    #                 rec_ln_pec = data['rec_ln_pec'])
    #     return model

    @classmethod
    def build(cls,line):
        if line not in atomic_data_lookup:
            raise KeyError('{} is not a valid atomic line identifier'.format(line))

        data = load(atomic_data_lookup[line])
        model = cls(ln_ne = data['ln_ne'],
                    ln_te = data['ln_te'],
                    exc_ln_pec = data['exc_ln_pec'],
                    rec_ln_pec = data['rec_ln_pec'],
                    emission_parameters = line_parameter_lookup[line])
        return model

    def gradient(self, te, ne, n_lower, n_upper):
        ln_te = log(te)
        ln_ne = log(ne)

        dE_dnu = exp(self.log_recomb_pec.ev(ln_ne, ln_te))
        dE_dnl = exp(self.log_excite_pec.ev(ln_ne, ln_te))
        R = dE_dnu*n_upper
        X = dE_dnl*n_lower

        dR_dne = self.log_recomb_pec.ev(ln_ne, ln_te, dx=1)
        dR_dTe = self.log_recomb_pec.ev(ln_ne, ln_te, dy=1)

        dX_dne = self.log_excite_pec.ev(ln_ne, ln_te, dx=1)
        dX_dTe = self.log_excite_pec.ev(ln_ne, ln_te, dy=1)

        dE_dne = (R*dR_dne + X*dX_dne)/ne
        dE_dTe = (R*dR_dTe + X*dX_dTe)/te

        partials = dict(zip(self.emission_parameters, [dE_dTe, dE_dne, dE_dnl, dE_dnu]))
        return partials

    def emission_and_gradient(self, te, ne, n_lower, n_upper):
        ln_te = log(te)
        ln_ne = log(ne)

        dE_dnu = exp(self.log_recomb_pec.ev(ln_ne, ln_te))
        dE_dnl = exp(self.log_excite_pec.ev(ln_ne, ln_te))
        R = dE_dnu*n_upper
        X = dE_dnl*n_lower

        dR_dne = self.log_recomb_pec.ev(ln_ne, ln_te, dx=1)
        dR_dTe = self.log_recomb_pec.ev(ln_ne, ln_te, dy=1)

        dX_dne = self.log_excite_pec.ev(ln_ne, ln_te, dx=1)
        dX_dTe = self.log_excite_pec.ev(ln_ne, ln_te, dy=1)

        dE_dne = (R*dR_dne + X*dX_dne)/ne
        dE_dTe = (R*dR_dTe + X*dX_dTe)/te

        partials = dict(zip(self.emission_parameters, [dE_dTe, dE_dne, dE_dnl, dE_dnu]))
        return R+X, partials


class AdasHydrogenMolecularModel(object):
    """
    Specific implementation for a general Balmer line
    """
    def __init__(self, model_line_dic=None, alpha_line_dic=None, emission_parameters=None):
        """
        Builds splines to be used by emissivity calculation methods based on a model including an effective molecular
            contribution to the emission
        Different model used for when this model's specified line is the alpha line vs other lines
        Stores information on the spline data limits.
        Data dictionaries containing arrays:
            (1D for electron density and electron temperature; 2D for excitation, recombination and effective
            molecular PEC data)
        Parameters:
        :param (dictionary) model_line_dic: Stores data for this model's specified line each as numpy arrays
        :param (dictionary) alpha_line_dic: Stores data for the alpha line each as numpy arrays
        :param list emission_parameters: strings corresponding to the parameters associated with this emission model
        """
        # store member variables
        self.emission_parameters = emission_parameters

        # get the ne mesh for the atomic specified line
        log_ne_atom_line_mesh, __ = meshgrid(model_line_dic['ln_ne_atm'], model_line_dic['ln_te_atm'], indexing='ij')
        # get the ne mesh for the atomic alpha line
        log_ne_atom_alpha_mesh, __ = meshgrid(alpha_line_dic['ln_ne_atm'], alpha_line_dic['ln_te_atm'], indexing='ij')

        # build the splines for the atomic line
        self.log_recomb = spl2d(model_line_dic['ln_ne_atm'], model_line_dic['ln_te_atm'],
                                model_line_dic['ln_rec_atm'] + 2*log_ne_atom_line_mesh)
        self.log_excite = spl2d(model_line_dic['ln_ne_atm'], model_line_dic['ln_te_atm'],
                                model_line_dic['ln_exc_atm'] + log_ne_atom_line_mesh)

        # Also need to build the alpha-atom splines (rec and exc), specified line-mol spline and alpha-mol spline
        self.log_recomb_alpha = spl2d(alpha_line_dic['ln_ne_atm'], alpha_line_dic['ln_te_atm'],
                                      alpha_line_dic['ln_rec_atm'] + 2*log_ne_atom_alpha_mesh)
        self.log_excite_alpha = spl2d(alpha_line_dic['ln_ne_atm'], alpha_line_dic['ln_te_atm'],
                                      alpha_line_dic['ln_exc_atm'] + log_ne_atom_alpha_mesh)
        self.log_molec_pec = spl2d(model_line_dic['ln_ne_mol'], model_line_dic['ln_te_mol'],
                                   model_line_dic['ln_eff_mol'])
        self.log_molec_pec_alpha = spl2d(alpha_line_dic['ln_ne_mol'], alpha_line_dic['ln_te_mol'],
                                         alpha_line_dic['ln_eff_mol'])

        # store the limit of the spline data; take the limiting value across the four different arrays
        limiting_min_ln_ne = max([model_line_dic['ln_ne_atm'].min(), alpha_line_dic['ln_ne_atm'].min(),
                                  model_line_dic['ln_ne_mol'].min(), alpha_line_dic['ln_ne_mol'].min()])
        limiting_max_ln_ne = min([model_line_dic['ln_ne_atm'].max(), alpha_line_dic['ln_ne_atm'].max(),
                                  model_line_dic['ln_ne_mol'].max(), alpha_line_dic['ln_ne_mol'].max()])
        limiting_min_ln_te = max([model_line_dic['ln_te_atm'].min(), alpha_line_dic['ln_te_atm'].min(),
                                  model_line_dic['ln_te_mol'].min(), alpha_line_dic['ln_te_mol'].min()])
        limiting_max_ln_te = min([model_line_dic['ln_te_atm'].max(), alpha_line_dic['ln_te_atm'].max(),
                                  model_line_dic['ln_te_mol'].max(), alpha_line_dic['ln_te_mol'].max()])
        self.ne_lims = [exp(limiting_min_ln_ne), exp(limiting_max_ln_ne)]
        self.te_lims = [exp(limiting_min_ln_te), exp(limiting_max_ln_te)]

    def __call__(self, te, ne, n0, q_mol):
        """
        Finds and returns the total emissivity
        Parameters
        :param float te: the electron temperature
        :param float ne: the electron density
        :param float n0: the neutral density
        :param float q_mol: the ratio of alpha-line emission due to molecular effects to that from atomic effects
        Returns
        :return float: The total emissivity for this model's line according to these parameters
        """
        e_rec = self.recombination(te, ne)
        e_exc = self.excitation(te, ne, n0)
        e_mol = self.molecular(te, ne, n0, q_mol)
        return e_mol + e_rec + e_exc

    def recombination(self, te, ne):
        """
        Finds and returns the recombination emissivity from atomic processes
        Parameters
        :param float te: the electron temperature
        :param float ne: the electron density
        Returns
        :return float: The emissivity due to atomic recombination processes for this model's line according to these
            parameters
        """
        ln_te = log(te)
        ln_ne = log(ne)
        return exp(self.log_recomb.ev(ln_ne, ln_te))

    def excitation(self, te, ne, n0):
        """
        Finds and returns the excitation emissivity from atomic processes
        Parameters
        :param float te: the electron temperature
        :param float ne: the electron density
        :param float n0: the neutral density
        Returns
        :return float: The emissivity due to atomic excitation processes for this model's line according to these
            parameters
        """
        ln_te = log(te)
        ln_ne = log(ne)
        return exp(self.log_excite.ev(ln_ne, ln_te)) * n0

    def molecular(self, te, ne, n0, q_mol):
        """
        Finds and returns the emissivity from molecular processes.
        Note: this is only to be called for non alpha lines since for alpha lines the molecular relevant splines do not
            need to be made.
        Parameters
        :param float te: the electron temperature
        :param float ne: the electron density
        :param float n0: the neutral density
        :param float q_mol: the ratio of alpha-line emission due to molecular effects to that from atomic effects
        Returns
        :return float: The emissivity due to molecular processes for this model's line according to these parameters
        """
        ln_te = log(te)
        ln_ne = log(ne)

        # find for alpha
        alpha_e_rec = exp(self.log_recomb_alpha.ev(ln_ne, ln_te))
        alpha_e_exc = exp(self.log_excite_alpha.ev(ln_ne, ln_te)) * n0
        # for specific line
        ln_n_pec_mol = self.log_molec_pec.ev(ln_ne, ln_te)
        ln_alpha_pec_mol = self.log_molec_pec_alpha.ev(ln_ne, ln_te)
        return q_mol * (alpha_e_rec + alpha_e_exc) * exp(ln_n_pec_mol - ln_alpha_pec_mol)

    def log_model(self, ln_te, ln_ne, ln_n0, ln_q_mol):
        """
        mimics __call__ method but uses log parameter values.
        Parameters
        :param float ln_te: the natural log of the electron temperature
        :param float ln_ne: the natural log of the electron density
        :param float ln_n0: the natural log of the neutral density
        :param float ln_q_mol: the natural log of the ratio of alpha-line emission due to molecular effects to that
            from atomic effects
        Returns
        :return float: The total emissivity for this model's line according to these parameters
        """
        e_exc = exp(self.log_excite.ev(ln_ne, ln_te) + ln_n0)
        e_rec = exp(self.log_recomb.ev(ln_ne, ln_te))

        ln_mol_pec = self.log_molec_pec.ev(ln_ne, ln_te)
        # find for alpha
        e_rec_alpha = exp(self.log_recomb_alpha.ev(ln_ne, ln_te))
        e_exc_alpha = exp(self.log_excite_alpha.ev(ln_ne, ln_te) + ln_n0)
        ln_mol_pec_alpha = self.log_molec_pec_alpha.ev(ln_ne, ln_te)
        e_mol = (e_rec_alpha + e_exc_alpha) * exp(ln_q_mol + ln_mol_pec - ln_mol_pec_alpha)
        return e_mol + e_rec + e_exc

    def recombination_pec(self, ln_ne, ln_te):
        """
        Finds the atomic recombination PEC value for this model's line (for testing pec data purposes)
        Parameters
        :param float ln_te: the natural log of the electron temperature
        :param float ln_ne: the natural log of the electron density
        Returns
        :return float: the atomic recombination PEC value for specified log(ne), log(te) pair for this model
        """
        return exp(self.log_recomb.ev(ln_ne, ln_te) - 2 * ln_ne)

    def excitation_pec(self, ln_ne, ln_te):
        """
        Finds the atomic excitation PEC value for this model's line (for testing pec data purposes)
        Parameters
        :param float ln_te: the natural log of the electron temperature
        :param float ln_ne: the natural log of the electron density
        Returns
        :return float: the atomic excitation PEC value for specified log(ne), log(te) pair for this model
        """
        return exp(self.log_excite.ev(ln_ne, ln_te) - ln_ne)

    def molecular_pec(self, ln_ne, ln_te):
        """
        Finds the effective molecular PEC value for this model's line (for testing pec data purposes)
        Parameters
        :param float ln_te: the natural log of the electron temperature
        :param float ln_ne: the natural log of the electron density
        Returns
        :return float: the effective molecular PEC value for specified log(ne), log(te) pair for this model
        """
        return exp(self.log_molec_pec.ev(ln_ne, ln_te))

    def alpha_molecular_pec(self, ln_ne, ln_te):
        """
        Finds the effective molecular PEC value for the alpha line (for testing pec data purposes)
        Parameters
        :param float ln_te: the natural log of the electron temperature
        :param float ln_ne: the natural log of the electron density
        Returns
        :return float: the effective molecular PEC value for specified log(ne), log(te) pair for the alpha line
        """
        return exp(self.log_molec_pec_alpha.ev(ln_ne, ln_te))

    @staticmethod
    def organise_data(line_str):
        """
        Takes the string name for the desired line, finds and loads the corresponding data and stores in an organised
            dictionary
        Data includes electron density and electron temperature grids for the atomic (atm) PEC grids as well as PEC
            values for the recombination (rec) and excitation (exc). Data also includes the electron density and
            electron temperature grids for the molecular (mol) PEC grids as well as PEC values for the effective (eff)
            molecular contribution
        All data are natural log values.
        Mimics naming within '.npz' files within '/model_data/'
        Parameters
        :param string line_str: the name identifier within atomic_data_lookup of
            '\model_data\__init__.py' corresponding to the data (atomic and molecular) for the desired line
        Returns
        :return: dictionary corresponding to organised density, temperature and pec values for the desired line.
        """
        # get the data
        line_atom_data = load(atomic_data_lookup[line_str])
        line_molc_data = load(molecular_data_lookup[line_str])

        # organise
        line_data = {
            'ln_ne_atm': line_atom_data['ln_ne'],
            'ln_te_atm': line_atom_data['ln_te'],
            'ln_rec_atm': line_atom_data['rec_ln_pec'],
            'ln_exc_atm': line_atom_data['exc_ln_pec'],
            'ln_ne_mol': line_molc_data['ln_ne'],
            'ln_te_mol': line_molc_data['ln_te'],
            'ln_eff_mol': line_molc_data['eff_mol_ln_pec'],
        }
        return line_data

    @classmethod
    def load(cls, path, is_alpha_line, emission_parameters, organising_function=None):
        """
        Used to build the emission model with molecular contribution from a path.
        The path should lead to an .npz file with data for the specified line.
        If the data is not in the required format (a dictionary with specific keys - see 'organise_data' method),
        an organising function must provided to transform it to the required format.
        Parameters:
        :param (string) path: the path for the specified line's data file
        :param (bool) is_alpha_line: whether this model is desired to be for the alpha line or not
        :param (list) emission_parameters: each element is a string corresponding to the parameters used in thi model's
            call method (order must be obeyed)
        :param (function/None) organising_function: if the data extracted from the path is not in the desired format,
            an organising function can be provided which receives the data from the path and returns an organised
            dictionary that can be used in this class (see 'organise_data' method for desired format)
        Returns:
        :return: a built emission model for a specified Balmer line
        """
        alpha_line_str = 'D_alpha'
        alpha_line_data = cls.organise_data(alpha_line_str)

        model_line_data = load(path)
        if organising_function is not None:
            model_line_data = organising_function(model_line_data)

        # if it is an alpha line, use the daughter class with specific methods
        if is_alpha_line:
            model = AdasHydrogenMolecularAlphaModel(
                model_line_dic=model_line_data,
                emission_parameters=emission_parameters
            )
        else:
            model = cls(model_line_dic=model_line_data,
                        alpha_line_dic=alpha_line_data,
                        emission_parameters=emission_parameters
                        )
        return model

    @classmethod
    def build(cls, model_line_str):
        """
        Used to build the molecular model as specified by model_line_str.
        From the emissivity model, information on both the atomic and molecular PEC values for the specified line is
             required as well as atomic and molecular PEC values for the alpha line.
        The necessary data is extracted and organised fo be used to initialise the class
        Parameters:
            :param (str) model_line_str: the atomic line identifier for this specified model
        """
        # NAMING:
        alpha_line_str = 'D_alpha'

        # CHECKS:
        # ensure that there is data for each of the required lines
        required_lines = [model_line_str, alpha_line_str]
        for required_line in required_lines:
            if required_line not in atomic_data_lookup:
                raise KeyError('No atomic data for atomic line identifier {}'.format(required_line))
            if required_line not in molecular_data_lookup:
                raise KeyError('No molecular data for atomic line identifier {}'.format(required_line))

        # DATA
        # extract
        model_line_data = cls.organise_data(model_line_str)
        alpha_line_data = cls.organise_data(alpha_line_str)

        # find if desired model is for alpha line(in which case can be calculated differently)
        is_alpha_line = model_line_str == alpha_line_str
        # if it is an alpha line, use the daughter class with specific methods
        if is_alpha_line:
            model = AdasHydrogenMolecularAlphaModel(
                model_line_dic=model_line_data,
                emission_parameters=line_parameter_molecular_lookup[model_line_str]
            )
        else:
            model = cls(model_line_dic=model_line_data,
                        alpha_line_dic=alpha_line_data,
                        emission_parameters=line_parameter_molecular_lookup[model_line_str]
                        )
        return model

    def gradient(self, te, ne, n0, q_mol):
        """
        Initially written for readability, can be refactored for speed
        dy_dx = dlny_dlnx * (y/x)
        Parameter Naming format:
            par: {te, ne, n0, q_mol}, the electron temperature, electron density, neutral density and molecular q ratio
            ln_[par] is the log of the parameter value
        Variable Naming format:
            0:
            ln = log value
            1:
            n = nth line
            alpha = alpha line
            2:
            e = emissivity
            pec = PEC
            3:
            rec = recombination
            exc = excitation
            mol = molecular

            convention: [(0)]_[1]_[2]_[3] so eg. ln_n_e_rec is the log of the nth line emissivity for recombination
            differentials: d_[var]_by_d_[par] where var is a variable and par is a parameter (shown above)
        """
        ln_te = log(te)
        ln_ne = log(ne)

        # Spline values
        n_e_rec = exp(self.log_recomb.ev(ln_ne, ln_te))
        n_e_exc = exp(self.log_excite.ev(ln_ne, ln_te)) * n0

        # Differentials (in log log space)
        d_ln_n_e_rec_by_d_ln_ne = self.log_recomb.ev(ln_ne, ln_te, dx=1)
        d_ln_n_e_rec_by_d_ln_te = self.log_recomb.ev(ln_ne, ln_te, dy=1)

        d_ln_n_e_exc_by_d_ln_ne = self.log_excite.ev(ln_ne, ln_te, dx=1)
        d_ln_n_e_exc_by_d_ln_te = self.log_excite.ev(ln_ne, ln_te, dy=1)

        # Differentials (in normal space)
        d_n_e_rec_by_d_ne = d_ln_n_e_rec_by_d_ln_ne * (n_e_rec / ne)
        d_n_e_rec_by_d_te = d_ln_n_e_rec_by_d_ln_te * (n_e_rec / te)

        d_n_e_exc_by_d_ne = d_ln_n_e_exc_by_d_ln_ne * (n_e_exc / ne)
        d_n_e_exc_by_d_te = d_ln_n_e_exc_by_d_ln_te * (n_e_exc / te)

        # Spline values
        alpha_e_rec = exp(self.log_recomb_alpha.ev(ln_ne, ln_te))
        alpha_e_exc = exp(self.log_excite_alpha.ev(ln_ne, ln_te)) * n0
        alpha_pec_mol = exp(self.log_molec_pec_alpha.ev(ln_ne, ln_te))
        n_pec_mol = exp(self.log_molec_pec.ev(ln_ne, ln_te))

        # Ratios
        n_alpha_r_mol = n_pec_mol / alpha_pec_mol

        # Differentials (in log log space)
        d_ln_n_pec_mol_by_d_ln_ne = self.log_molec_pec.ev(ln_ne, ln_te, dx=1)
        d_ln_n_pec_mol_by_d_ln_te = self.log_molec_pec.ev(ln_ne, ln_te, dy=1)

        d_ln_alpha_pec_mol_by_d_ln_ne = self.log_molec_pec_alpha.ev(ln_ne, ln_te, dx=1)
        d_ln_alpha_pec_mol_by_d_ln_te = self.log_molec_pec_alpha.ev(ln_ne, ln_te, dy=1)

        d_ln_alpha_e_rec_by_d_ln_ne = self.log_recomb_alpha.ev(ln_ne, ln_te, dx=1)
        d_ln_alpha_e_rec_by_d_ln_te = self.log_recomb_alpha.ev(ln_ne, ln_te, dy=1)

        d_ln_alpha_e_exc_by_d_ln_ne = self.log_excite_alpha.ev(ln_ne, ln_te, dx=1)
        d_ln_alpha_e_exc_by_d_ln_te = self.log_excite_alpha.ev(ln_ne, ln_te, dy=1)

        # Differentials (in normal space)
        d_n_alpha_r_mol_by_ne = (d_ln_n_pec_mol_by_d_ln_ne - d_ln_alpha_pec_mol_by_d_ln_ne) * n_pec_mol / (
                    alpha_pec_mol * ne)
        d_n_alpha_r_mol_by_te = (d_ln_n_pec_mol_by_d_ln_te - d_ln_alpha_pec_mol_by_d_ln_te) * n_pec_mol / (
                alpha_pec_mol * te)

        d_alpha_e_rec_by_d_ne = d_ln_alpha_e_rec_by_d_ln_ne * (alpha_e_rec / ne)
        d_alpha_e_rec_by_d_te = d_ln_alpha_e_rec_by_d_ln_te * (alpha_e_rec / te)

        d_alpha_e_exc_by_d_ne = d_ln_alpha_e_exc_by_d_ln_ne * (alpha_e_exc / ne)
        d_alpha_e_exc_by_d_te = d_ln_alpha_e_exc_by_d_ln_te * (alpha_e_exc / te)

        # final differentials
        d_n_e_by_d_te = q_mol * (n_alpha_r_mol * d_alpha_e_rec_by_d_te +
                                 alpha_e_rec * d_n_alpha_r_mol_by_te +
                                 n_alpha_r_mol * d_alpha_e_exc_by_d_te +
                                 alpha_e_exc * d_n_alpha_r_mol_by_te) + d_n_e_rec_by_d_te + d_n_e_exc_by_d_te
        d_n_e_by_d_ne = q_mol * (n_alpha_r_mol * d_alpha_e_rec_by_d_ne +
                                 alpha_e_rec * d_n_alpha_r_mol_by_ne +
                                 n_alpha_r_mol * d_alpha_e_exc_by_d_ne +
                                 alpha_e_exc * d_n_alpha_r_mol_by_ne) + d_n_e_rec_by_d_ne + d_n_e_exc_by_d_ne
        d_n_e_by_d_n0 = q_mol * n_alpha_r_mol * alpha_e_exc / n0 + n_e_exc / n0
        d_n_e_by_d_q = (alpha_e_rec + alpha_e_exc) * (n_pec_mol / alpha_pec_mol)

        # Emissivity gradient wrt temperature, density and emitter neutral density
        return {'Te': d_n_e_by_d_te, 'ne': d_n_e_by_d_ne, 'n0': d_n_e_by_d_n0, 'q_mol': d_n_e_by_d_q}

    def emission_and_gradient(self, te, ne, n0, q_mol):
        emissivity = self(te, ne, n0, q_mol)
        gradient_dictionary = self.gradient(te, ne, n0, q_mol)
        return emissivity, gradient_dictionary


class AdasHydrogenMolecularAlphaModel(AdasHydrogenMolecularModel):
    """
    Specific implementation for the Balmer-alpha line.
    """
    def __init__(self, model_line_dic=None, emission_parameters=None):
        """
        Builds splines to be used by emissivity calculation methods based on a model including an effective molecular
            contribution to the emission
        Stores information on the spline data limits.
        Data dictionaries containing arrays:
            (1D for electron density and electron temperature; 2D for excitation, recombination and effective
            molecular PEC data)
        Parameters:
        :param (dictionary) model_line_dic: Stores data for this model's specified line (alpha)
        :param list emission_parameters: strings corresponding to the parameters associated with this emission model
        """
        # get the ne mesh for the atomic specified line
        log_ne_atom_line_mesh, __ = meshgrid(model_line_dic['ln_ne_atm'], model_line_dic['ln_te_atm'], indexing='ij')
        # store member variables
        self.emission_parameters = emission_parameters

        # build the splines for the atomic line
        self.log_recomb = spl2d(model_line_dic['ln_ne_atm'], model_line_dic['ln_te_atm'],
                                model_line_dic['ln_rec_atm'] + 2*log_ne_atom_line_mesh)
        self.log_excite = spl2d(model_line_dic['ln_ne_atm'], model_line_dic['ln_te_atm'],
                                model_line_dic['ln_exc_atm'] + log_ne_atom_line_mesh)

        # store the limit of the spline data
        self.ne_lims = [exp(model_line_dic['ln_ne_atm'].min()), exp(model_line_dic['ln_ne_atm'].max())]
        self.te_lims = [exp(model_line_dic['ln_te_atm'].min()), exp(model_line_dic['ln_te_atm'].max())]

    def __call__(self, te, ne, n0, q_mol):
        """
        Finds and returns the total emissivity
        Parameters
        :param float te: the electron temperature
        :param float ne: the electron density
        :param float n0: the neutral density
        :param float q_mol: the ratio of alpha-line emission due to molecular effects to that from atomic effects
        Returns
        :return float: The total emissivity for the alpha line according to these parameters
        """
        e_rec = self.recombination(te, ne)
        e_exc = self.excitation(te, ne, n0)
        return (e_rec + e_exc) * (1 + q_mol)

    def gradient(self, te, ne, n0, q_mol):
        """
        Initially written for readability, can be refactored for speed
        dy_dx = dlny_dlnx * (y/x)
        Parameter Naming format:
            par: {te, ne, n0, q_mol}, the electron temperature, electron density, neutral density and molecular q ratio
            ln_[par] is the log of the parameter value
        Variable Naming format:
            0:
            ln = log value
            1:
            n = nth line (alpha)
            2:
            e = emissivity
            pec = PEC
            3:
            rec = recombination
            exc = excitation
            mol = molecular

            convention: [(0)]_[1]_[2]_[3] so eg. ln_n_e_rec is the log of the nth line emissivity for recombination
            differentials: d_[var]_by_d_[par] where var is a variable and par is a parameter (shown above)
        """
        ln_te = log(te)
        ln_ne = log(ne)

        # Spline values
        n_e_rec = exp(self.log_recomb.ev(ln_ne, ln_te))
        n_e_exc = exp(self.log_excite.ev(ln_ne, ln_te)) * n0

        # Differentials (in log log space)
        d_ln_n_e_rec_by_d_ln_ne = self.log_recomb.ev(ln_ne, ln_te, dx=1)
        d_ln_n_e_rec_by_d_ln_te = self.log_recomb.ev(ln_ne, ln_te, dy=1)

        d_ln_n_e_exc_by_d_ln_ne = self.log_excite.ev(ln_ne, ln_te, dx=1)
        d_ln_n_e_exc_by_d_ln_te = self.log_excite.ev(ln_ne, ln_te, dy=1)

        # Differentials (in normal space)
        d_n_e_rec_by_d_ne = d_ln_n_e_rec_by_d_ln_ne * (n_e_rec / ne)
        d_n_e_rec_by_d_te = d_ln_n_e_rec_by_d_ln_te * (n_e_rec / te)

        d_n_e_exc_by_d_ne = d_ln_n_e_exc_by_d_ln_ne * (n_e_exc / ne)
        d_n_e_exc_by_d_te = d_ln_n_e_exc_by_d_ln_te * (n_e_exc / te)

        d_n_e_by_d_te = (1 + q_mol) * (d_n_e_rec_by_d_te + d_n_e_exc_by_d_te)
        d_n_e_by_d_ne = (1 + q_mol) * (d_n_e_rec_by_d_ne + d_n_e_exc_by_d_ne)
        d_n_e_by_d_n0 = (1 + q_mol) * n_e_exc / n0
        d_n_e_by_d_q = n_e_rec + n_e_exc

        # Emissivity gradient wrt temperature, density and emitter neutral density
        return {'Te': d_n_e_by_d_te, 'ne': d_n_e_by_d_ne, 'n0': d_n_e_by_d_n0, 'q_mol': d_n_e_by_d_q}

    def log_model(self, ln_te, ln_ne, ln_n0, ln_q_mol):
        """
        mimics __call__ method but uses log parameter values.
        Parameters
        :param float ln_te: the natural log of the electron temperature
        :param float ln_ne: the natural log of the electron density
        :param float ln_n0: the natural log of the neutral density
        :param float ln_q_mol: the natural log of the ratio of alpha-line emission due to molecular effects to that
            from atomic effects
        Returns
        :return float: The total emissivity for this model's line according to these parameters
        """
        e_exc = exp(self.log_excite.ev(ln_ne, ln_te) + ln_n0)
        e_rec = exp(self.log_recomb.ev(ln_ne, ln_te))

        return (e_rec + e_exc) * (1 + exp(ln_q_mol))

    def molecular(self, te, ne, n0, q_mol):
        """
        Finds and returns the emissivity from molecular processes.
        Note: this is not required to be explicitly called for the __call__ method. It is provided for testing purposes.
        Parameters
        :param float te: the electron temperature
        :param float ne: the electron density
        :param float n0: the neutral density
        :param float q_mol: the ratio of alpha-line emission due to molecular effects to that from atomic effects
        Returns
        :return float: The emissivity due to molecular processes for this model's line according to these parameters
        """

        e_rec = self.recombination(te, ne)
        e_exc = self.excitation(te, ne, n0)
        return (e_rec + e_exc) * q_mol

    def molecular_pec(self, ln_ne, ln_te):
        """
        No effective molecular PEC value is required for this model's line and so an error should be raised
        (for testing pec data purposes)
        """
        msg = 'No method is provided for finding the effective molecular pec value for the alpha line' \
              ' since it is not required.'
        raise NameError(msg)
