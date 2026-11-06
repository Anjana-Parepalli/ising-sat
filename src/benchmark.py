from time import perf_counter

import numpy as np

from src.annealing import simulated_annealing
from src.ising import spins_to_binary
from src.sat import count_violations

def benchmark_annealing(
    clauses, n_variables, h, J, C,
    runs=20, sweeps=1000,
    initial_temp=5.0, final_temp=0.01,
):
    energies = []
    durations = []
    successes = 0

    for seed in range(runs):
        start = perf_counter()

        spins, energy, _ = simulated_annealing(
            h, J, C,
            sweeps=sweeps,
            initial_temp=initial_temp,
            final_temp=final_temp,
            seed=seed,
        )

        duration = perf_counter() - start
        assignment = spins_to_binary(spins)[:n_variables]

        energies.append(energy)
        durations.append(duration)

        if count_violations(clauses, assignment) == 0:
            successes += 1

    return {
        "success_rate": successes / runs,
        "mean_energy": float(np.mean(energies)),
        "best_energy": float(np.min(energies)),
        "mean_runtime": float(np.mean(durations)),
        "energies": energies,
    }