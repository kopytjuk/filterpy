import numpy as np
from numpy.testing import assert_equal

from filterpy.common.model_conversion import constacc2constvel, constvel2constacc


def test_constvel2constacc_2d():

    state_vector = np.array([10, 30, -10, -30])

    state_vector_acc = constvel2constacc(state_vector, dim=2)

    assert_equal(state_vector_acc[:2], state_vector[:2])
    assert_equal(state_vector_acc[3:-1], state_vector[2:])
    assert state_vector_acc[2] == 0.0
    assert state_vector_acc[5] == 0.0


def test_constvel2constacc_3d():

    state_vector = np.array([10, 30, 10, 0, -10, -30])

    state_vector_acc = constvel2constacc(state_vector, dim=3)

    assert_equal(state_vector_acc[:2], state_vector[:2])
    assert state_vector_acc[2] == 0.0
    assert state_vector_acc[5] == 0.0
    assert state_vector_acc[8] == 0.0


def test_constacc2constvel_2d():

    state_vector = np.array([10, 30, 0, -10, -30, 0])

    state_vector_vel = constacc2constvel(state_vector, dim=2)

    assert_equal(state_vector_vel[:2], state_vector[:2])
    assert_equal(state_vector_vel[2:], state_vector[3:-1])
