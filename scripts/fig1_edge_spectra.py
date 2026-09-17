"""Figure 1: frequency-resolved structural edge roles."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D


blue = "#245A9A"
orange = "#E07A1F"
green = "#238B45"
grey = "#D5D5D5"
dark = "#202020"

plt.rc(
    "font",
    family="serif",
    serif=["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    size=10,
)
plt.rc("mathtext", fontset="cm")


GROUP_SIZE = 15
P_WITHIN = 0.55
P_MIDDLE = 0.08
P_OUTER = 0.015
SEED = 190
ETA_FRACTION = 0.03
TOP_FRACTION = 0.10

# Maxima of the ensemble enrichment curves in Fig. 2.
FREQUENCIES = [0.032, 0.123, 0.486]

COLORS = {
    "within group": blue,
    "between groups": orange,
    "between supergroups": green,
}


def edge_class(graph, edge):
    block_u = graph.nodes[edge[0]]["block"]
    block_v = graph.nodes[edge[1]]["block"]
    if block_u == block_v:
        return "within group"
    if block_u // 2 == block_v // 2:
        return "between groups"
    return "between supergroups"


def sample_graph():
    probabilities = np.full((4, 4), P_OUTER)
    np.fill_diagonal(probabilities, P_WITHIN)
    probabilities[0, 1] = probabilities[1, 0] = P_MIDDLE
    probabilities[2, 3] = probabilities[3, 2] = P_MIDDLE

    graph = nx.stochastic_block_model(
        [GROUP_SIZE] * 4,
        probabilities,
        seed=SEED,
    )

    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        for left, right in zip(components[:-1], components[1:]):
            graph.add_edge(next(iter(left)), next(iter(right)))

    return graph


def fixed_layout(graph):
    centers = {
        0: (-0.72, 0.53),
        1: (-0.72, -0.53),
        2: (0.72, 0.53),
        3: (0.72, -0.53),
    }

    position = {}
    for block in range(4):
        nodes = [
            node
            for node in graph.nodes()
            if graph.nodes[node]["block"] == block
        ]
        angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        angles += 0.18 * (block % 2)
        center_x, center_y = centers[block]

        for node, angle in zip(nodes, angles):
            position[node] = (
                center_x + 0.24 * np.cos(angle),
                center_y + 0.24 * np.sin(angle),
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

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.35))

for panel, (ax, frequency) in enumerate(zip(axes, FREQUENCIES)):
    nx.draw_networkx_edges(
        graph,
        position,
        edgelist=edges,
        edge_color=grey,
        width=0.32,
        alpha=0.32,
        ax=ax,
    )

    values = scores[:, panel]
    number = max(1, int(round(TOP_FRACTION * len(edges))))
    selected_indices = np.argpartition(values, -number)[-number:]
    selected_values = values[selected_indices]

    low = selected_values.min()
    high = selected_values.max()
    scale = (selected_values - low) / max(high - low, 1e-14)

    for kind, color in COLORS.items():
        local_indices = [
            j
            for j, index in enumerate(selected_indices)
            if edge_class(graph, edges[index]) == kind
        ]
        if not local_indices:
            continue

        selected_edges = [
            edges[selected_indices[j]]
            for j in local_indices
        ]
        widths = 1.0 + 1.6 * scale[local_indices]

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
        node_size=21,
        node_color="white",
        edgecolors=dark,
        linewidths=0.55,
        ax=ax,
    )

    ax.set_title(
        rf"$\Omega/\lambda_{{\max}}={frequency:.3f}$",
        fontsize=10,
        pad=3,
    )
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.92, 0.92)
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

legend = [
    Line2D([0], [0], color=color, linewidth=2, label=kind)
    for kind, color in COLORS.items()
]
fig.legend(
    handles=legend,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8.5,
    bbox_to_anchor=(0.5, -0.01),
)

fig.subplots_adjust(
    left=0.01,
    right=0.995,
    bottom=0.13,
    top=0.85,
    wspace=0.06,
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
