import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_ising_network(h, J):
    graph = nx.Graph()

    for i in range(len(h)):
        graph.add_node(i, field=h[i])

    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            if not np.isclose(J[i, j], 0):
                graph.add_edge(
                    i, j, weight=J[i, j]
                )

    fig, ax = plt.subplots(figsize=(7, 5))
    pos = nx.spring_layout(graph, seed=42)

    colors = [
        "royalblue" if graph[u][v]["weight"] > 0
        else "tomato"
        for u, v in graph.edges()
    ]

    widths = [
        1 + 4 * abs(graph[u][v]["weight"])
        for u, v in graph.edges()
    ]

    nx.draw_networkx(
        graph,
        pos,
        ax=ax,
        node_color="lightblue",
        edge_color=colors,
        width=widths,
        node_size=900,
        labels={
            i: f"s{i+1}" for i in graph.nodes()
        },
    )

    ax.set_title("Ising Spin Interaction Network")
    ax.axis("off")

    return fig