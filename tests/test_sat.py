from src.sat import (
    evaluate_clause,
    count_violations,
    brute_force_solve,
)


def test_clause():
    assert evaluate_clause([1, -2], [0, 0])
    assert not evaluate_clause([1, 2], [0, 0])


def test_violations():
    clauses = [[1, 2], [-1, 3]]

    assert count_violations(clauses, [0, 0, 0]) == 1
    assert count_violations(clauses, [1, 0, 0]) == 1
    assert count_violations(clauses, [1, 0, 1]) == 0


def test_brute_force():
    clauses = [[1, 2], [-1, 3]]

    solutions = brute_force_solve(clauses, 3)

    assert len(solutions) == 4
    assert (1, 0, 1) in solutions
    assert (0, 0, 0) not in solutions