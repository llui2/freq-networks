"""Figure 1: frequency-resolved edge roles in a hierarchical network."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


navy = "#132A8A"
orange = "#E07A1F"
red = "#C92A2A"
grey = "#C9C9C9"
dark = "#202020"

plt.rc("font", family="serif", serif=["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"], size=10)
plt.rc("mathtext", fontset="cm")


GROUP_SIZE = 15
P_WITHIN = 0.55
P_MIDDLE = 0.08
P_OUTER = 0.015
SEED = 190
ETA_FRACTION = 0.03
TOP_FRACTION = 0.05
FREQUENCIES = [0.05, 0.15, 0.45]
COLORS = [navy, orange, red]


def sample_graph():
    probabilities = np.full((4, 4), P_OUTER)
    np.fill_diagonal(probabilities, P_WITHIN)
    probabilities[0, 1] = probabilities[1, 0] = P_MIDDLE
    probabilities[2, 3] = probabilities[3, 2] = P_MIDDLE
    return nx.stochastic_block_model(
        [GROUP_SIZE] * 4,
        probabilities,
        seed=SEED,
    )


def fixed_layout(graph):
    centers = [-1.82, -0.84, 0.84, 1.82]
    position = {}
    for block in range(4):
        nodes = [
            node
            for node in graph.nodes()
            if graph.nodes[node]["block"] == block
        ]
        angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        angles += 0.28 * (block % 2)
        for node, angle in zip(nodes, angles):
            position[node] = (
                centers[block] + 0.27 * np.cos(angle),
                0.52 * np.sin(angle),
            )
    return position


def edge_scores(graph, frequencies):
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    lambda_max = eigenvalues[-1]
    eta = ETA_FRACTION * lambda_max

    edges = list(graph.edges())
    scores = np.zeros((len(edges), len(frequencies)))
    for edge_index, (u, v) in enumerate(edges):
        incidence = np.zeros(len(nodes))
        incidence[index[u]] = 1.0
        incidence[index[v]] = -1.0
        coefficients = eigenvectors.T @ incidence
        for frequency_index, frequency in enumerate(frequencies):
            omega = frequency * lambda_max
            scores[edge_index, frequency_index] = (
                eta
                / np.pi
                * np.sum(
                    coefficients**2
                    / ((eigenvalues - omega) ** 2 + eta**2)
                )
            )
    return edges, scores


graph = sample_graph()
position = fixed_layout(graph)
edges, scores = edge_scores(graph, FREQUENCIES)

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.55))

for panel, (ax, frequency, color) in enumerate(
    zip(axes, FREQUENCIES, COLORS)
):
    nx.draw_networkx_edges(
        graph,
        position,
        edgelist=edges,
        edge_color=grey,
        width=0.35,
        alpha=0.28,
        ax=ax,
    )

    values = scores[:, panel]
    number = max(1, int(round(TOP_FRACTION * len(edges))))
    selected_indices = np.argpartition(values, -number)[-number:]
    selected_edges = [edges[index] for index in selected_indices]
    selected_values = values[selected_indices]
    low = selected_values.min()
    high = selected_values.max()
    scale = (selected_values - low) / max(high - low, 1e-14)
    widths = 1.1 + 1.9 * scale

    nx.draw_networkx_edges(
        graph,
        position,
        edgelist=selected_edges,
        edge_color=color,
        width=widths,
        alpha=0.92,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        position,
        node_size=24,
        node_color="white",
        edgecolors=dark,
        linewidths=0.65,
        ax=ax,
    )

    ax.set_title(
        rf"$\Omega/\lambda_{{\max}}={frequency:.2f}$",
        fontsize=10,
        pad=4,
    )
    ax.set_xlim(-2.23, 2.23)
    ax.set_ylim(-0.78, 0.78)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(
        -0.02,
        1.02,
        chr(ord("a") + panel),
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontname="DejaVu Sans",
    )

fig.subplots_adjust(
    left=0.015,
    right=0.995,
    bottom=0.03,
    top=0.86,
    wspace=0.07,
)

root = Path(__file__).resolve().parents[1]
figure_dir = root / "paper" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(
    figure_dir / "fig1_edge_spectra.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
)
fig.savefig(
    figure_dir / "fig1_edge_spectra.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
)
print(f"Saved figure to {figure_dir / 'fig1_edge_spectra.pdf'}")
