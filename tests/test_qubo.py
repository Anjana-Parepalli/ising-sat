from itertools import product

import numpy as np
import pytest

from src.sat import count_violations
from src.qubo import sat_to_qubo, qubo_energy
from src.ising import (
    qubo_to_ising,
    ising_energy,
    binary_to_spins,
    spins_to_binary,
)
from src.annealing import simulated_annealing

@pytest.mark.parametrize("clauses", [
    [[1, 2, 3]],
    [[-1, 2, -3]],
    [[1, -2, 3], [-1, 2, -3]],
    [[1, 1, 1]],
    [[1, -1, 2]],
    [[1, 2, 3], [-1, -2, -3], [1, -2]],
])
def test_3sat_energy_equivalence(clauses):
    n = 3
    Q, offset = sat_to_qubo(clauses, n)

    n_aux = Q.shape[0] - n

    for original in product([0, 1], repeat=n):
        energies = []

        for auxiliary in product([0, 1], repeat=n_aux):
            state = original + auxiliary
            energies.append(qubo_energy(Q, offset, state))

        expected = count_violations(clauses, original)

        assert np.isclose(min(energies), expected)

def test_3sat_ising_equivalence():
    clauses = [[1, 2, 3], [-1, -2, -3]]
    Q, offset = sat_to_qubo(clauses, 3)
    h, J, C = qubo_to_ising(Q, offset)

    for state in product([0, 1], repeat=Q.shape[0]):
        spins = binary_to_spins(state)

        assert np.isclose(
            qubo_energy(Q, offset, state),
            ising_energy(h, J, C, spins),
        )

def test_3sat_end_to_end():
    clauses = [[1, 2, 3], [-1, 2, -3], [1, -2, 3]]

    Q, offset = sat_to_qubo(clauses, 3)
    h, J, C = qubo_to_ising(Q, offset)

    spins, energy, history = simulated_annealing(
        h, J, C,
        sweeps=2000,
        seed=42,
    )

    complete_state = spins_to_binary(spins)

    # Only the first three variables belong to the SAT formula.
    original_assignment = complete_state[:3]

    assert np.isclose(energy, 0.0)
    assert count_violations(clauses, original_assignment) == 0