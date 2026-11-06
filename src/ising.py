import numpy as np

def qubo_to_ising(Q, offset=0.0):
    """
    Convert an upper-triangular QUBO into an Ising model.

    QUBO:
        E(x) = x.T @ Q @ x + offset

    Ising:
        H(s) = h.T @ s + sum(J[i,j] * s[i] * s[j]) + C

    Convention:
        x = (1 - s) / 2
    """

    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]

    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)

    C = float(offset)

    for i in range(n):
        h[i] -= Q[i, i] / 2
        C += Q[i, i] / 2

    for i in range(n):
        for j in range(i + 1, n):
            coefficient = Q[i, j]

            J[i, j] = coefficient / 4

            h[i] -= coefficient / 4
            h[j] -= coefficient / 4

            C += coefficient / 4

    return h, J, C

def ising_energy(h, J, C, spins):
    """Calculate the energy of a spin configuration."""

    s = np.asarray(spins, dtype=float)

    return float(
        np.dot(h, s)
        + s @ J @ s
        + C
    )

def spins_to_binary(spins):
    """Convert Ising spins (-1, +1) to binary values (0, 1)."""

    s = np.asarray(spins, dtype=int)

    return ((1 - s) // 2).astype(int)

def binary_to_spins(assignment):
    """Convert binary values (0, 1) to Ising spins (+1, -1)."""

    x = np.asarray(assignment, dtype=int)

    return 1 - 2 * x