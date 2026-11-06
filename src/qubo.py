import numpy as np

def sat_to_qubo(clauses, n_variables):
    """
    Convert SAT clauses of length 1, 2, or 3 into QUBO.

    Returns an upper-triangular Q and an energy offset.

    Original variables occupy indices 0 to n_variables-1.
    Each 3-literal clause introduces one auxiliary variable.
    """
    if n_variables < 0:
        raise ValueError("n_variables must be nonnegative")

    for clause in clauses:
        if not 1 <= len(clause) <= 3:
            raise ValueError("Clauses must contain 1 to 3 literals")
        if any(
            not isinstance(lit, int)
            or lit == 0
            or abs(lit) > n_variables
            for lit in clause
        ):
            raise ValueError("Invalid literal")

    n_aux = sum(len(clause) == 3 for clause in clauses)
    n_total = n_variables + n_aux

    Q = np.zeros((n_total, n_total), dtype=float)
    offset = 0.0
    next_aux = n_variables

    def false_factor(literal):
        """Return (constant, coefficient, variable index)."""
        index = abs(literal) - 1

        if literal > 0:
            return (1.0, -1.0, index)

        return (0.0, 1.0, index)

    def add_linear(factor, weight=1.0):
        nonlocal offset

        a, b, i = factor
        offset += weight * a
        Q[i, i] += weight * b

    def add_product(first, second, weight=1.0):
        nonlocal offset

        a, b, i = first
        c, d, j = second

        offset += weight * a * c
        Q[i, i] += weight * b * c
        Q[j, j] += weight * a * d

        if i == j:
            Q[i, i] += weight * b * d
        else:
            Q[min(i, j), max(i, j)] += weight * b * d

    for clause in clauses:
        factors = [false_factor(lit) for lit in clause]

        if len(factors) == 1:
            add_linear(factors[0])

        elif len(factors) == 2:
            add_product(factors[0], factors[1])

        else:
            u, v, w = factors

            z_index = next_aux
            next_aux += 1
            z = (0.0, 1.0, z_index)

            # z represents u * v.
            # Original clause penalty becomes z * w.
            add_product(z, w)

            # Enforce z = u*v using a quadratic penalty.
            penalty_weight = 2.0

            add_product(u, v, penalty_weight)
            add_product(u, z, -2 * penalty_weight)
            add_product(v, z, -2 * penalty_weight)
            add_linear(z, 3 * penalty_weight)

    return Q, offset

def qubo_energy(Q, offset, assignment):
    """Evaluate QUBO energy for a complete binary state."""
    x = np.asarray(assignment, dtype=float)
    return float(x @ Q @ x + offset)