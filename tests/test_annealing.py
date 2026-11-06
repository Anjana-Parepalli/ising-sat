from itertools import product

import numpy as np

from src.qubo import sat_to_qubo
from src.ising import (
    qubo_to_ising,
    ising_energy,
    spins_to_binary,
)
from src.sat import count_violations
from src.annealing import simulated_annealing

def test_spin_flip_energy():
    clauses = [[1, 2], [-1, 3]]

    Q, offset = sat_to_qubo(clauses, 3)
    h, J, C = qubo_to_ising(Q, offset)

    couplings = J + J.T

    for configuration in product([-1, 1], repeat=3):
        spins = np.array(configuration)

        for i in range(3):
            flipped = spins.copy()
            flipped[i] *= -1

            actual_delta = (
                ising_energy(h, J, C, flipped)
                - ising_energy(h, J, C, spins)
            )

            predicted_delta = (
                -2 * spins[i]
                * (h[i] + np.dot(couplings[i], spins))
            )

            assert np.isclose(actual_delta, predicted_delta)

def test_annealing_finds_solution():
    clauses = [[1, 2], [-1, 3]]

    Q, offset = sat_to_qubo(clauses, 3)
    h, J, C = qubo_to_ising(Q, offset)

    spins, energy, history = simulated_annealing(
        h, J, C,
        sweeps=1000,
        seed=42,
    )

    assignment = spins_to_binary(spins)

    assert np.isclose(energy, 0.0)
    assert count_violations(clauses, assignment) == 0
    assert len(history) == 1001

def test_reproducibility():
    h = np.array([0.5, -0.3])
    J = np.array([[0.0, 0.2], [0.0, 0.0]])

    result1 = simulated_annealing(
        h, J, sweeps=100, seed=123
    )

    result2 = simulated_annealing(
        h, J, sweeps=100, seed=123
    )

    assert np.array_equal(result1[0], result2[0])
    assert np.isclose(result1[1], result2[1])
    assert np.allclose(result1[2], result2[2])