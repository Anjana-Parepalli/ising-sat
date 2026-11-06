from itertools import product

def evaluate_clause(clause, assignment):
    """Check whether a single SAT clause is satisfied."""

    for literal in clause:
        index = abs(literal) - 1
        value = assignment[index]

        if literal < 0:
            value = 1 - value

        if value == 1:
            return True

    return False

def count_violations(clauses, assignment):
    """Count how many clauses are violated."""

    return sum(
        not evaluate_clause(clause, assignment)
        for clause in clauses
    )


def brute_force_solve(clauses, n_variables):
    """Find all satisfying assignments by exhaustive search."""

    solutions = []

    for assignment in product([0, 1], repeat=n_variables):
        if count_violations(clauses, assignment) == 0:
            solutions.append(assignment)

    return solutions