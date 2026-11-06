from itertools import product

import numpy as np

from src.qubo import sat_to_qubo, qubo_energy
from src.ising import (
    qubo_to_ising,
    ising_energy,
    binary_to_spins,
    spins_to_binary,
)

def test_qubo_ising_energy_equivalence():
    clauses = [[1, 2], [-1, 3]]

    Q, offset = sat_to_qubo(clauses, 3)
    h, J, C = qubo_to_ising(Q, offset)

    for assignment in product([0, 1], repeat=3):
        x = np.array(assignment)
        spins = binary_to_spins(x)

        E_qubo = qubo_energy(Q, offset, x)
        E_ising = ising_energy(h, J, C, spins)

        assert np.isclose(E_qubo, E_ising)

def test_binary_spin_roundtrip():
    for assignment in product([0, 1], repeat=4):
        x = np.array(assignment)

        spins = binary_to_spins(x)
        recovered = spins_to_binary(spins)

        assert np.array_equal(x, recovered)

def test_known_ising_coefficients():
    clauses = [[1, 2], [-1, 3]]

    Q, offset = sat_to_qubo(clauses, 3)
    h, J, C = qubo_to_ising(Q, offset)

    expected_h = np.array([0.0, 0.25, 0.25])

    expected_J = np.array([
        [0.0, 0.25, -0.25],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    assert np.allclose(h, expected_h)
    assert np.allclose(J, expected_J)
    assert np.isclose(C, 0.5)