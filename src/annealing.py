import numpy as np

def simulated_annealing(
    h,
    J,
    C=0.0,
    initial_temp=5.0,
    final_temp=0.01,
    sweeps=1000,
    seed=None,
):
    """
    Minimize an Ising Hamiltonian using simulated annealing.

    One sweep attempts n spin flips, where n is the
    number of spins.

    Returns:
        best_spins: Lowest-energy configuration found
        best_energy: Its energy
        history: Energy recorded after each sweep
    """

    rng = np.random.default_rng(seed)

    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)

    n = len(h)
    couplings = J + J.T

    # Initialize random spins.
    spins = rng.choice([-1, 1], size=n)

    # Calculate the initial energy.
    energy = (
        np.dot(h, spins)
        + spins @ J @ spins
        + C
    )

    best_spins = spins.copy()
    best_energy = float(energy)

    history = [float(energy)]

    for sweep in range(sweeps):

        # Geometric cooling schedule.
        fraction = sweep / max(sweeps - 1, 1)

        temperature = initial_temp * (
            final_temp / initial_temp
        ) ** fraction

        # Attempt n spin flips per sweep.
        for _ in range(n):
            i = rng.integers(n)

            local_field = (
                h[i] + np.dot(couplings[i], spins)
            )

            delta_energy = -2 * spins[i] * local_field

            # Metropolis acceptance rule.
            if (
                delta_energy <= 0
                or rng.random() < np.exp(-delta_energy / temperature)
            ):
                spins[i] *= -1
                energy += delta_energy

                if energy < best_energy:
                    best_energy = float(energy)
                    best_spins = spins.copy()

        history.append(float(energy))

    return best_spins, best_energy, history