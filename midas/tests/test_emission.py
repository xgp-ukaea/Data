import pytest
import unittest
from numpy import array, zeros, allclose, log, load
from math import isclose
from midas.emission import construct_emission_model

"""
    Non-tested helper methods
"""


def finite_difference(func=None, x0=None, delta=1e-5):
    grad = zeros(x0.size)
    for i in range(x0.size):
        x1 = x0.copy()
        x2 = x0.copy()
        dx = x0[i]*delta

        x1[i] -= dx
        x2[i] += dx
        f1 = func(*x1)
        f2 = func(*x2)
        grad[i] = 0.5*(f2-f1)/dx
    return grad


"""
    Testing PEC Expected Values
"""


@pytest.mark.parametrize(
    "line_n, data_type, data_file",
    [
        (3, 'atomic_recombination', 'adas_atomic_pec_test_data.npz'),
        (4, 'atomic_recombination', 'adas_atomic_pec_test_data.npz'),
        (5, 'atomic_recombination', 'adas_atomic_pec_test_data.npz'),
        (6, 'atomic_recombination', 'adas_atomic_pec_test_data.npz'),
        (7, 'atomic_recombination', 'adas_atomic_pec_test_data.npz'),
        (3, 'atomic_excitation',    'adas_atomic_pec_test_data.npz'),
        (4, 'atomic_excitation',    'adas_atomic_pec_test_data.npz'),
        (5, 'atomic_excitation',    'adas_atomic_pec_test_data.npz'),
        (6, 'atomic_excitation',    'adas_atomic_pec_test_data.npz'),
        (7, 'atomic_excitation',    'adas_atomic_pec_test_data.npz'),
        # todo get test case from Kevin
        # line_n = 4 since molecular data only needed for lines n=4 onward
        # (4, 'alpha_effective_molecular', ''),
        # (4, 'effective_molecular',       ''),
        # (5, 'effective_molecular',       ''),
        # (6, 'effective_molecular',       ''),
        # (7, 'effective_molecular',       '')
    ]
)
def test_expected_pecs(line_n, data_type, data_file):
    line_index = line_n-3
    lines = ['D_alpha', 'D_beta', 'D_gamma', 'D_delta', 'D_epsilon']
    line = lines[line_index]
    # extract data
    checked_data = load(data_file)
    checked_data = checked_data[data_type][line_index]  # just keep the data needed for this test

    model = construct_emission_model(line, is_include_mol_effects=True)
    ln_nes = log(checked_data[:, 0])
    ln_tes = log(checked_data[:, 1])
    checked_pecs = checked_data[:, 2]
    if data_type == 'atomic_recombination':
        found_pecs = model.recombination_pec(ln_nes, ln_tes)
    elif data_type == 'atomic_excitation':
        found_pecs = model.excitation_pec(ln_nes, ln_tes)
    elif data_type == 'effective_molecular':
        found_pecs = model.molecular_pec(ln_nes, ln_tes)
    elif data_type == 'alpha_effective_molecular':
        found_pecs = model.alpha_molecular_pec(ln_nes, ln_tes)
    else:
        msg = 'Unrecognised data_type: {}'.format(data_type)
        raise ValueError(msg)

    # Error Buffer
    max_error_percentage = 2
    rtol = max_error_percentage/100
    atol = 0.  # the absolute difference

    # look for agreement:
    assert allclose(checked_pecs, found_pecs, rtol=rtol, atol=atol)


"""
    Emmisivity Expected Value Tests
"""


@pytest.mark.parametrize(
    # todo get test case from Kevin
    "line, params, expected_emissivities",
    [
        ('D_alpha', array([3.5, 2e19, 1.6e18, 20]), [5e22, 7.e18, 2.5e21, 5.e22]),
        ('D_delta', array([3.5, 2e19, 1.6e18, 20]), [1.3e20, 8.e17, 5.e18, 1.3e20])
    ]
)
def test_hydrogen_mol_model_expected_emissivities(line, params, expected_emissivities):
    """
    Verify model's emissivity aligns with expected emissivity values for certain parameter combination
    params order: te, ne, n0, q_mol
    expected_emissivities order: total emissivity, recombination emissivity, excitation emissivity,
     molecular contribution emissivity
    """
    spline = construct_emission_model(line, is_include_mol_effects=True)

    emissivity = spline(*params)
    recombination_emissivity = spline.recombination(*params[0:2])
    excitation_emissivity = spline.excitation(*params[0:3])
    molecular_emissivity = spline.molecular(*params)

    # Expected values
    expected_emissivity = expected_emissivities[0]
    expected_recombination_emissivity = expected_emissivities[1]
    expected_excitation_emissivity = expected_emissivities[2]
    expected_molecular_emissivity = expected_emissivities[3]

    # Error Buffer
    max_error_percentage = 10
    rel_tol = max_error_percentage/100
    abs_tol = 0.

    assert isclose(emissivity, expected_emissivity, abs_tol=abs_tol, rel_tol=rel_tol)
    assert isclose(recombination_emissivity, expected_recombination_emissivity, abs_tol=abs_tol, rel_tol=rel_tol)
    assert isclose(excitation_emissivity, expected_excitation_emissivity, abs_tol=abs_tol, rel_tol=rel_tol)
    assert isclose(molecular_emissivity, expected_molecular_emissivity, abs_tol=abs_tol, rel_tol=rel_tol)


@pytest.mark.parametrize(
    "line, params",
    [
        ('D_alpha', array([3, 2e19, 1e17, 1e16])),
        ('D_beta',  array([10, 5e19, 5e16, 1e17]))
    ]
)
def test_atomic_and_molecular_model_agreement(line, params):
    """
    Check that the atomic component of the model with molecular contribution agrees with the standard atomic model
    params order: Te, ne, n0, q_mol
    """
    pure_atomic_model = construct_emission_model(line)
    with_molec_model = construct_emission_model(line, is_include_mol_effects=True)
    assert isclose(pure_atomic_model(*params[:3]),
                   with_molec_model.recombination(*params[:2]) + with_molec_model.excitation(*params[:3]))


@pytest.mark.parametrize(
    "line, params, is_molecular_model",
    [
        ('D_alpha', array([3, 2e19, 1e17]), False),
        ('D_alpha', array([3, 2e19, 1e17, 1e16]), True),
        ('D_beta',  array([10, 5e19, 5e16]), False),
        ('D_beta',  array([10, 5e19, 5e16, 1e17]), True)
    ]
)
def test_log_model_agreement(line, params, is_molecular_model):
    """
    Check that the log model call agrees in value with the standard model call
    """
    model = construct_emission_model(line, is_include_mol_effects=is_molecular_model)
    assert isclose(model(*params), model.log_model(*log(params)))


"""
    Gradient Calculation Tests
"""


@pytest.mark.parametrize(
    "line, params",
    [
        ('He_6680', array([3, 2e19, 1e17, 1e16])),
        ('He_7283', array([10, 5e19, 5e16, 1e17]))
    ]
)
def test_impurity_model_gradient(line, params):
    """
    params order: Te, ne, n0, impurity density
    """
    spline = construct_emission_model(line)

    fd_gradient = finite_difference(func=spline, x0=params)

    emission_1 = spline(*params)
    emission_2, grad_dict_1 = spline.emission_and_gradient(*params)
    grad_dict_2 = spline.gradient(*params)

    grad_array_1 = array([v for v in grad_dict_1.values()])
    grad_array_2 = array([v for v in grad_dict_2.values()])

    assert emission_1 == emission_2
    assert (grad_array_1 == grad_array_2).all()
    assert allclose(grad_array_1, fd_gradient)


@pytest.mark.parametrize(
    "line, params",
    [
        ('D_alpha',   array([1.,  2e17, 1e19])),
        ('D_beta',    array([40., 1e20, 1e16])),
        ('D_gamma',   array([3.5, 2e19, 1.6e18])),
        ('D_delta',   array([0.5, 1e17, 1e17])),
        ('D_epsilon', array([20., 1e19, 1e18])),
    ]
)
def test_hydrogen_model_gradient(line, params):
    """
    params order: Te, ne, n0
    """
    spline = construct_emission_model(line)

    fd_gradient = finite_difference(func=spline, x0=params)

    emission_1 = spline(*params)
    emission_2, grad_dict_1 = spline.emission_and_gradient(*params)
    grad_dict_2 = spline.gradient(*params)

    grad_array_1 = array([v for v in grad_dict_1.values()])
    grad_array_2 = array([v for v in grad_dict_2.values()])

    assert emission_1 == emission_2
    assert (grad_array_1 == grad_array_2).all()
    assert allclose(grad_array_1, fd_gradient)


@pytest.mark.parametrize(
    "line, params",
    [
        ('D_alpha',   array([1.,  2e17, 1e19, 30])),
        ('D_beta',    array([40., 1e20, 1e16, 0.01])),
        ('D_gamma',   array([5.,  1e19, 1e18, 3])),
        ('D_delta',   array([0.5, 1e17, 1e17, 10])),
        ('D_epsilon', array([20., 1e19, 1e18, 0.5])),
    ]
)
def test_hydrogen_mol_model_gradient(line, params):
    """
    params order: Te, ne, n0, q_mol
    """
    spline = construct_emission_model(line, is_include_mol_effects=True)
    fd_gradient = finite_difference(func=spline, x0=params)
    emission, grad_dict = spline.emission_and_gradient(*params)
    grad_array = array([v for v in grad_dict.values()])
    assert allclose(grad_array, fd_gradient)


"""
    Usage tests
"""


@pytest.mark.parametrize(
    "line",
    [
        'bad_string',
        0.,
    ]
)
def test_fails_on_bad_construct_emission_model(line):
    """
    Try to construct an emission model without a valid string identifier
    """
    with pytest.raises(KeyError):
        construct_emission_model(line)
    with pytest.raises(KeyError):
        construct_emission_model(line, is_include_mol_effects=True)


if __name__ == '__main__':

    unittest.main()
