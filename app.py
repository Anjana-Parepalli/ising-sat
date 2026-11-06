import ast
from itertools import product
from time import perf_counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from src.sat import count_violations, brute_force_solve
from src.qubo import sat_to_qubo, qubo_energy
from src.ising import (
    qubo_to_ising,
    ising_energy,
    spins_to_binary,
)
from src.annealing import simulated_annealing

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IsingSAT",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ IsingSAT")
st.caption(
    "A Classical Ising Machine for Solving Boolean "
    "Satisfiability Problems through QUBO Optimization"
)

st.markdown(
    """
    **Computational pipeline:**

    SAT → QUBO → Ising Hamiltonian →
    Simulated Annealing → Solution Verification
    """
)

# ============================================================
# 2. SAT PROBLEM PRESETS
# ============================================================

PRESETS = {
    "Satisfiable 2-SAT": (
        [[1, 2], [-1, 3]],
        3,
    ),
    "Satisfiable 3-SAT": (
        [[1, 2, 3], [-1, 2, -3], [1, -2, 3]],
        3,
    ),
    "Unsatisfiable 2-SAT": (
        [[1], [-1]],
        1,
    ),
    "Unsatisfiable 3-SAT": (
        [
            [a, b, c]
            for a, b, c in product([1, -1], repeat=3)
        ],
        3,
    ),
}

# ============================================================
# 3. SIDEBAR CONFIGURATION
# ============================================================

st.sidebar.header("Problem Configuration")

preset = st.sidebar.selectbox(
    "SAT Problem",
    list(PRESETS.keys()) + ["Custom"],
)

if preset == "Custom":
    raw = st.sidebar.text_area(
        "Clauses (Python list format)",
        value="[[1, 2, 3], [-1, 2, -3]]",
        height=120,
    )

    n_variables = int(
        st.sidebar.number_input(
            "Number of Boolean variables",
            min_value=1,
            max_value=20,
            value=3,
        )
    )

    try:
        clauses = ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError):
        st.error("Invalid SAT clause format.")
        st.stop()

else:
    clauses, n_variables = PRESETS[preset]

# Validate input before compilation.
if (
    not isinstance(clauses, (list, tuple))
    or not all(
        isinstance(clause, (list, tuple))
        and 1 <= len(clause) <= 3
        and all(
            type(literal) is int
            and literal != 0
            and abs(literal) <= n_variables
            for literal in clause
        )
        for clause in clauses
    )
):
    st.error(
        "Each clause must contain 1–3 valid integer literals."
    )
    st.stop()


st.sidebar.header("Annealing Parameters")

sweeps = st.sidebar.slider(
    "Annealing sweeps",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
)

initial_temp = st.sidebar.slider(
    "Initial temperature",
    min_value=0.1,
    max_value=10.0,
    value=5.0,
    step=0.1,
)

final_temp = st.sidebar.slider(
    "Final temperature",
    min_value=0.001,
    max_value=1.0,
    value=0.01,
    step=0.001,
)

seed = int(
    st.sidebar.number_input(
        "Random seed",
        min_value=0,
        value=42,
    )
)

benchmark_runs = st.sidebar.slider(
    "Benchmark trials",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
)

if final_temp >= initial_temp:
    st.error(
        "Final temperature must be lower than initial temperature."
    )
    st.stop()


# ============================================================
# 4. COMPILE SAT -> QUBO -> ISING
# ============================================================

try:
    Q, offset = sat_to_qubo(clauses, n_variables)
    h, J, C = qubo_to_ising(Q, offset)
except (ValueError, TypeError) as error:
    st.error(f"Compilation failed: {error}")
    st.stop()

n_spins = len(h)
n_aux = n_spins - n_variables

if n_spins > 100:
    st.error(
        "This interactive demo supports at most 100 spins."
    )
    st.stop()

signature = (
    tuple(tuple(clause) for clause in clauses),
    n_variables,
    sweeps,
    initial_temp,
    final_temp,
    seed,
)

benchmark_signature = signature + (benchmark_runs,)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def plot_matrix(matrix, title, labels=None):
    """Create a heatmap of a numerical matrix."""

    fig, ax = plt.subplots(figsize=(7, 5))

    limit = max(float(np.max(np.abs(matrix))), 1e-9)

    image = ax.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )

    ax.set_title(title)
    ax.set_xlabel("Variable index")
    ax.set_ylabel("Variable index")

    if labels is not None and len(labels) <= 20:
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)

    fig.colorbar(image, ax=ax, label="Coefficient")
    fig.tight_layout()

    return fig


def plot_ising_network(h, J, n_original):
    """Visualize spins and their pairwise interactions."""

    graph = nx.Graph()

    for i in range(len(h)):
        graph.add_node(i)

    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            if not np.isclose(J[i, j], 0):
                graph.add_edge(
                    i,
                    j,
                    weight=float(J[i, j]),
                )

    fig, ax = plt.subplots(figsize=(9, 6))

    positions = nx.spring_layout(
        graph,
        seed=42,
    )

    node_colors = [
        "skyblue" if i < n_original else "orange"
        for i in graph.nodes()
    ]

    edge_colors = [
        "royalblue" if graph[u][v]["weight"] > 0
        else "tomato"
        for u, v in graph.edges()
    ]

    edge_widths = [
        1 + 2 * abs(graph[u][v]["weight"])
        for u, v in graph.edges()
    ]

    labels = {
        i: (
            f"x{i + 1}"
            if i < n_original
            else f"z{i - n_original + 1}"
        )
        for i in graph.nodes()
    }

    nx.draw_networkx(
        graph,
        positions,
        ax=ax,
        labels=labels,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        node_size=850,
        font_size=10,
    )

    ax.set_title("Ising Spin Interaction Network")
    ax.axis("off")
    fig.tight_layout()

    return fig


def run_benchmark(
    clauses,
    n_variables,
    h,
    J,
    C,
    runs,
    sweeps,
    initial_temp,
    final_temp,
):
    """Run independent annealing trials."""

    records = []

    for trial in range(runs):
        start = perf_counter()

        spins, energy, _ = simulated_annealing(
            h,
            J,
            C,
            initial_temp=initial_temp,
            final_temp=final_temp,
            sweeps=sweeps,
            seed=trial,
        )

        elapsed = perf_counter() - start

        assignment = spins_to_binary(spins)[:n_variables]
        violations = count_violations(clauses, assignment)

        records.append({
            "Trial": trial + 1,
            "Energy": float(energy),
            "Violations": int(violations),
            "SAT Found": violations == 0,
            "Runtime (ms)": elapsed * 1000,
        })

    return pd.DataFrame(records)


def exact_qubo_minimum(Q, offset, max_spins=18):
    """Enumerate small QUBOs to find their exact minimum."""

    n = Q.shape[0]

    if n > max_spins:
        return None

    minimum = float("inf")

    for state in product([0, 1], repeat=n):
        energy = qubo_energy(Q, offset, state)
        minimum = min(minimum, energy)

    return minimum


# ============================================================
# 6. PROBLEM OVERVIEW
# ============================================================

st.header("1. SAT Problem")

st.code(str(clauses), language="python")

a, b, c, d = st.columns(4)

a.metric("Boolean Variables", n_variables)
b.metric("Clauses", len(clauses))
c.metric("Auxiliary Variables", n_aux)
d.metric("Total Ising Spins", n_spins)

st.info(
    "Auxiliary variables are introduced when converting "
    "3-SAT clauses into quadratic energy functions."
)


# ============================================================
# 7. RUN ISING MACHINE
# ============================================================

st.header("2. Ising Machine Optimization")

if st.button(
    "Run Ising Machine",
    type="primary",
):
    start = perf_counter()

    spins, energy, history = simulated_annealing(
        h,
        J,
        C,
        initial_temp=initial_temp,
        final_temp=final_temp,
        sweeps=sweeps,
        seed=seed,
    )

    elapsed = perf_counter() - start

    complete_state = spins_to_binary(spins)
    assignment = complete_state[:n_variables]

    violations = count_violations(
        clauses,
        assignment,
    )

    st.session_state["ising_result"] = {
        "signature": signature,
        "spins": spins,
        "complete_state": complete_state,
        "assignment": assignment,
        "energy": energy,
        "history": history,
        "violations": violations,
        "runtime": elapsed,
    }


result = st.session_state.get("ising_result")

if result is not None and result["signature"] == signature:

    violations = result["violations"]
    satisfied = len(clauses) - violations

    a, b, c, d = st.columns(4)

    a.metric(
        "Best Energy",
        f"{result['energy']:.4f}",
    )

    b.metric(
        "Satisfied Clauses",
        f"{satisfied}/{len(clauses)}",
    )

    c.metric(
        "Violations",
        violations,
    )

    d.metric(
        "Runtime",
        f"{result['runtime'] * 1000:.2f} ms",
    )

    if violations == 0:
        st.success(
            "SAT: A satisfying assignment was found and verified."
        )
    else:
        st.warning(
            "No satisfying assignment was found in this run. "
            "This does not prove that the problem is UNSAT."
        )

    st.subheader("Recovered Boolean Assignment")

    assignment_df = pd.DataFrame({
        "Variable": [
            f"x{i + 1}" for i in range(n_variables)
        ],
        "Value": result["assignment"],
    })

    st.dataframe(
        assignment_df,
        hide_index=True,
        use_container_width=True,
    )

    if n_aux > 0:
        with st.expander("Show Auxiliary Variables"):
            auxiliary_df = pd.DataFrame({
                "Auxiliary Variable": [
                    f"z{i + 1}" for i in range(n_aux)
                ],
                "Value": result["complete_state"][n_variables:],
            })

            st.dataframe(
                auxiliary_df,
                hide_index=True,
            )

    # --------------------------------------------------------
    # Energy convergence
    # --------------------------------------------------------

    st.subheader("Energy Convergence")

    history = np.asarray(result["history"])
    best_history = np.minimum.accumulate(history)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        history,
        label="Current Energy",
        alpha=0.65,
    )

    ax.plot(
        best_history,
        label="Best Energy So Far",
        linewidth=2,
    )

    ax.set_xlabel("Annealing Sweep")
    ax.set_ylabel("Energy")
    ax.set_title("Simulated Annealing Convergence")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Current energy may increase temporarily because "
        "the Metropolis algorithm sometimes accepts "
        "energy-increasing spin flips."
    )

    # --------------------------------------------------------
    # Verify energy consistency
    # --------------------------------------------------------

    st.subheader("Energy Consistency Check")

    qubo_e = qubo_energy(
        Q,
        offset,
        result["complete_state"],
    )

    ising_e = ising_energy(
        h,
        J,
        C,
        result["spins"],
    )

    col1, col2 = st.columns(2)

    col1.metric("QUBO Energy", f"{qubo_e:.6f}")
    col2.metric("Ising Energy", f"{ising_e:.6f}")

    if np.isclose(qubo_e, ising_e):
        st.success(
            "QUBO and Ising energies are mathematically consistent."
        )
    else:
        st.error("Energy mismatch detected!")


# ============================================================
# 8. MATHEMATICAL REPRESENTATIONS
# ============================================================

st.header("3. Mathematical Representations")

tab1, tab2, tab3, tab4 = st.tabs([
    "QUBO Matrix",
    "Ising Couplings",
    "Local Fields",
    "Spin Network",
])

labels = [
    f"x{i + 1}" if i < n_variables
    else f"z{i - n_variables + 1}"
    for i in range(n_spins)
]

with tab1:
    st.latex(r"E(x)=x^TQx+c")

    st.write(f"Constant offset: {offset:.4f}")

    st.dataframe(
        pd.DataFrame(
            Q,
            index=labels,
            columns=labels,
        ),
        use_container_width=True,
    )

    fig = plot_matrix(
        Q,
        "QUBO Coefficient Matrix",
        labels,
    )

    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.latex(
        r"H(s)=\sum_i h_i s_i"
        r"+\sum_{i<j}J_{ij}s_is_j+C"
    )

    st.write(f"Ising constant offset: {C:.4f}")

    st.dataframe(
        pd.DataFrame(
            J,
            index=labels,
            columns=labels,
        ),
        use_container_width=True,
    )

    fig = plot_matrix(
        J + J.T,
        "Ising Interaction Matrix",
        labels,
    )

    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fields_df = pd.DataFrame({
        "Spin": labels,
        "Local Field (h)": h,
    })

    st.dataframe(
        fields_df,
        hide_index=True,
        use_container_width=True,
    )

    st.bar_chart(
        fields_df.set_index("Spin")
    )

with tab4:
    st.write(
        "Blue nodes represent original Boolean variables. "
        "Orange nodes represent auxiliary variables."
    )

    st.write(
        "Blue edges represent positive couplings; "
        "red edges represent negative couplings."
    )

    fig = plot_ising_network(
        h,
        J,
        n_variables,
    )

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 9. EXACT SOLVER VERIFICATION
# ============================================================

st.header("4. Exact Solver Verification")

if n_variables <= 16:

    start = perf_counter()

    solutions = brute_force_solve(
        clauses,
        n_variables,
    )

    exact_runtime = perf_counter() - start

    a, b, c = st.columns(3)

    a.metric(
        "Exact SAT Status",
        "SAT" if solutions else "UNSAT",
    )

    b.metric(
        "Satisfying Assignments",
        len(solutions),
    )

    c.metric(
        "Exact Runtime",
        f"{exact_runtime * 1000:.2f} ms",
    )

    if solutions:
        st.success(
            "The exact solver confirms that the "
            "SAT formula is satisfiable."
        )

        st.write(
            "Example satisfying assignment:",
            solutions[0],
        )

    else:
        st.error(
            "UNSAT: Exhaustive enumeration confirms "
            "that no satisfying assignment exists."
        )

else:
    st.info(
        "Exact SAT enumeration is disabled for more "
        "than 16 original variables."
    )


# ============================================================
# 10. EXACT QUBO MINIMUM
# ============================================================

st.header("5. Exact Energy Baseline")

if n_spins <= 18:

    with st.spinner("Calculating exact minimum energy..."):
        exact_energy = exact_qubo_minimum(
            Q,
            offset,
        )

    st.metric(
        "Global Minimum QUBO Energy",
        f"{exact_energy:.4f}",
    )

    if result is not None and result["signature"] == signature:
        gap = result["energy"] - exact_energy

        st.metric(
            "Annealing Optimality Gap",
            f"{max(0.0, gap):.6f}",
        )

        if np.isclose(gap, 0.0):
            st.success(
                "The Ising machine reached the global "
                "minimum energy."
            )
        else:
            st.warning(
                "The Ising machine did not reach the "
                "global minimum in this run."
            )

else:
    exact_energy = None

    st.info(
        "Exact QUBO enumeration is disabled for more "
        "than 18 spins."
    )


# ============================================================
# 11. MULTIPLE-RUN BENCHMARKING
# ============================================================

st.header("6. Experimental Benchmarking")

st.write(
    "Simulated annealing is stochastic. "
    "We therefore evaluate it over multiple independent "
    "runs using different random seeds."
)

if st.button("Run Benchmark"):

    with st.spinner(
        f"Running {benchmark_runs} independent trials..."
    ):
        benchmark_df = run_benchmark(
            clauses,
            n_variables,
            h,
            J,
            C,
            runs=benchmark_runs,
            sweeps=sweeps,
            initial_temp=initial_temp,
            final_temp=final_temp,
        )

    st.session_state["benchmark_result"] = {
        "signature": benchmark_signature,
        "data": benchmark_df,
    }


benchmark_result = st.session_state.get("benchmark_result")

if (
    benchmark_result is not None
    and benchmark_result["signature"] == benchmark_signature
):

    df = benchmark_result["data"]

    success_rate = df["SAT Found"].mean()
    mean_energy = df["Energy"].mean()
    mean_runtime = df["Runtime (ms)"].mean()
    best_energy = df["Energy"].min()

    a, b, c, d = st.columns(4)

    a.metric(
        "SAT Success Rate",
        f"{success_rate:.1%}",
    )

    b.metric(
        "Mean Best Energy",
        f"{mean_energy:.3f}",
    )

    c.metric(
        "Best Energy Found",
        f"{best_energy:.3f}",
    )

    d.metric(
        "Mean Runtime",
        f"{mean_runtime:.2f} ms",
    )

    if exact_energy is not None:
        optimal_rate = np.mean(
            np.isclose(df["Energy"], exact_energy)
        )

        st.metric(
            "Global Optimum Hit Rate",
            f"{optimal_rate:.1%}",
        )

    st.subheader("Energy Distribution Across Trials")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        df["Trial"],
        df["Energy"],
    )

    if exact_energy is not None:
        ax.axhline(
            exact_energy,
            linestyle="--",
            color="red",
            label="Exact Minimum",
        )
        ax.legend()

    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Energy Found")
    ax.set_title("Annealing Results Across Independent Runs")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Detailed Trial Results")

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )

    csv_data = df.to_csv(index=False)

    st.download_button(
        "Download Benchmark CSV",
        data=csv_data,
        file_name="ising_benchmark.csv",
        mime="text/csv",
    )

    if not df["SAT Found"].any():
        st.warning(
            "No satisfying assignment was found in these trials. "
            "This alone does not establish UNSAT."
        )


# ============================================================
# 12. PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander("About This Project"):
    st.markdown(
        """
        ### IsingSAT: Classical Ising Machine Simulator

        This project investigates the use of Ising-based
        optimization for Boolean satisfiability problems.

        **Pipeline**

        1. Represent SAT formulas in conjunctive normal form.
        2. Convert logical clauses into QUBO energy penalties.
        3. Introduce auxiliary variables for 3-SAT clauses.
        4. Transform QUBO into an Ising Hamiltonian.
        5. Minimize the Hamiltonian using simulated annealing.
        6. Recover and verify Boolean assignments.
        7. Compare results against exact enumeration.
        8. Evaluate reliability using repeated trials.

        **Important distinction**

        This is a classical software simulation of an Ising
        optimization machine. It does not run on quantum hardware.

        **Limitations**

        - Simulated annealing does not guarantee global optimality.
        - SAT success is verified independently.
        - UNSAT requires an exact proof or exhaustive verification.
        - Exact enumeration scales exponentially.
        - Auxiliary variables increase the number of Ising spins.
        """
    )

st.caption(
    "IsingSAT | SAT → QUBO → Ising → Simulated Annealing "
    "| Classical Physics-Inspired Optimization"
)