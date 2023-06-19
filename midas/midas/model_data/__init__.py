import os

atomic_data_path = os.path.dirname(os.path.abspath(__file__))


if atomic_data_path.endswith('midas\model_data'):
    atomic_data_path += '\\'
else:
    atomic_data_path += '/'

# build paths dictionary
atomic_data_lookup = {
    'D_alpha'   : atomic_data_path + 'd_alpha_pec_data.npz',
    'D_beta'    : atomic_data_path + 'd_beta_pec_data.npz',
    'D_gamma'   : atomic_data_path + 'd_gamma_pec_data.npz',
    'D_delta'   : atomic_data_path + 'd_delta_pec_data.npz',
    'D_epsilon' : atomic_data_path + 'd_epsilon_pec_data.npz',
    'He_6680'   : atomic_data_path + 'He_6680_pec_data.npz',
    'He_7283'   : atomic_data_path + 'He_7283_pec_data.npz'
}

molecular_data_lookup = {
    'D_alpha'   : atomic_data_path + 'd_alpha_molecular_pec_data.npz',
    'D_beta'    : atomic_data_path + 'd_beta_molecular_pec_data.npz',
    'D_gamma'   : atomic_data_path + 'd_gamma_molecular_pec_data.npz',
    'D_delta'   : atomic_data_path + 'd_delta_molecular_pec_data.npz',
    'D_epsilon' : atomic_data_path + 'd_epsilon_molecular_pec_data.npz',
}

line_parameter_lookup = {
    'D_alpha'   : ['Te', 'ne', 'n0'],
    'D_beta'    : ['Te', 'ne', 'n0'],
    'D_gamma'   : ['Te', 'ne', 'n0'],
    'D_delta'   : ['Te', 'ne', 'n0'],
    'D_epsilon' : ['Te', 'ne', 'n0'],
    'He_6680'   : ['Te', 'ne', 'He0', 'He1'],
    'He_7283'   : ['Te', 'ne', 'He0', 'He1']
}

line_parameter_molecular_lookup = {
    'D_alpha'   : ['Te', 'ne', 'n0', 'Qmol'],
    'D_beta'    : ['Te', 'ne', 'n0', 'Qmol'],
    'D_gamma'   : ['Te', 'ne', 'n0', 'Qmol'],
    'D_delta'   : ['Te', 'ne', 'n0', 'Qmol'],
    'D_epsilon' : ['Te', 'ne', 'n0', 'Qmol']
}

hydrogen_model_lines = {'D_alpha', 'D_beta', 'D_gamma', 'D_delta', 'D_epsilon'}
impurity_model_lines = {'He_6680', 'He_7283'}
