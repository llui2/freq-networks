"""Figure 1: edge spectral fingerprints on a barbell graph."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


red = "#D62728"
blue = "#1F77B4"
grey = "#B7B7B7"
dark = "#222222"

plt.rc("font", family="Times", size=11)
plt.rc("mathtext", fontset="cm")


CLIQUE_SIZE = 8
ETA_FRACTION = 0.035
POINTS = 700


def eigensystem(graph):
    nodes = list(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    return nodes, laplacian, eigenvalues, eigenvectors


def edge_density(edge, nodes, eigenvalues, eigenvectors, omega, eta):
    index = {node: i for i, node in enumerate(nodes)}
    u, v = edge
    incidence = np.zeros(len(nodes))
    incidence[index[u]] = 1.0
    incidence[index[v]] = -1.0
    coefficients = eigenvectors.T @ incidence

    return (
        eta
        / np.pi
        * np.sum(
            coefficients[:, None] ** 2
            / (
                (eigenvalues[:, None] - omega[None, :]) ** 2
                + eta**2
            ),
            axis=0,
        )
    )


graph = nx.barbell_graph(CLIQUE_SIZE, 0)
bridge = (CLIQUE_SIZE - 1, CLIQUE_SIZE)

nodes, laplacian, eigenvalues, eigenvectors = eigensystem(graph)
lambda_max = eigenvalues[-1]
eta = ETA_FRACTION * lambda_max
omega = np.linspace(0.0, lambda_max, POINTS)

bridge_density = edge_density(
    bridge,
    nodes,
    eigenvalues,
    eigenvectors,
    omega,
    eta,
)

internal_edges = [
    edge
    for edge in graph.edges()
    if tuple(sorted(edge)) != bridge
]
internal_density = np.mean(
    [
        edge_density(
            edge,
            nodes,
            eigenvalues,
            eigenvectors,
            omega,
            eta,
        )
        for edge in internal_edges
    ],
    axis=0,
)


fig, (ax_a, ax_b) = plt.subplots(
    1,
    2,
    figsize=(6.6, 2.65),
    gridspec_kw={"width_ratios": [0.86, 1.30]},
)


# ---------------------------------------------------------------------------
# A. Graph
# ---------------------------------------------------------------------------

angles = np.linspace(
    np.pi / 2 + 0.40,
    np.pi / 2 + 0.40 + 2 * np.pi,
    CLIQUE_SIZE,
    endpoint=False,
)
position = {}
for i, angle in enumerate(angles):
    position[i] = (-1.15 + 0.55 * np.cos(angle), 0.55 * np.sin(angle))
    position[CLIQUE_SIZE + i] = (
        1.15 - 0.55 * np.cos(angle),
        0.55 * np.sin(angle),
    )

internal_left = [
    edge
    for edge in graph.edges()
    if edge[0] < CLIQUE_SIZE and edge != bridge
]
internal_right = [
    edge
    for edge in graph.edges()
    if edge[0] >= CLIQUE_SIZE and edge != bridge
]

nx.draw_networkx_edges(
    graph,
    position,
    edgelist=internal_left + internal_right,
    ax=ax_a,
    edge_color=grey,
    width=0.65,
    alpha=0.7,
)
nx.draw_networkx_edges(
    graph,
    position,
    edgelist=[bridge],
    ax=ax_a,
    edge_color=red,
    width=2.3,
)
nx.draw_networkx_nodes(
    graph,
    position,
    ax=ax_a,
    node_size=34,
    node_color="white",
    edgecolors=dark,
    linewidths=0.85,
)

ax_a.text(
    0.50,
    0.04,
    "bridge edge",
    color=red,
    fontsize=9,
    ha="center",
    va="bottom",
    transform=ax_a.transAxes,
)
ax_a.set_xlim(-2.0, 2.0)
ax_a.set_ylim(-0.95, 0.95)
ax_a.set_axis_off()


# ---------------------------------------------------------------------------
# B. Edge spectral density
# ---------------------------------------------------------------------------

x = omega / lambda_max

ax_b.plot(
    x,
    bridge_density,
    color=red,
    linewidth=1.8,
    label="bridge edge",
)
ax_b.plot(
    x,
    internal_density,
    color=blue,
    linewidth=1.8,
    label="within-clique edges",
)

positive_eigenvalues = eigenvalues[eigenvalues > 1e-10] / lambda_max
rug_y = 2.9e-3
for value in positive_eigenvalues:
    ax_b.plot(
        [value, value],
        [rug_y, 1.35 * rug_y],
        color="black",
        linewidth=0.55,
        alpha=0.38,
        clip_on=False,
    )

ax_b.set_yscale("log")
ax_b.set_xlim(0.0, 1.0)
ax_b.set_ylim(2.7e-3, 3.2)
ax_b.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax_b.set_xlabel(r"$\Omega/\lambda_{\max}$")
ax_b.set_ylabel(r"$\rho_e(\Omega,\eta)$")
ax_b.tick_params(direction="in", top=True, right=True)
ax_b.legend(
    loc="upper center",
    fontsize=8.5,
    frameon=False,
    ncol=1,
)
ax_b.spines["top"].set_linewidth(0.8)
ax_b.spines["right"].set_linewidth(0.8)
ax_b.spines["left"].set_linewidth(0.8)
ax_b.spines["bottom"].set_linewidth(0.8)


for label, ax in zip(("a", "b"), (ax_a, ax_b)):
    ax.text(
        -0.03,
        1.01,
        label,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontname="DejaVu Sans",
    )


fig.subplots_adjust(
    left=0.035,
    right=0.985,
    bottom=0.21,
    top=0.94,
    wspace=0.27,
)

root = Path(__file__).resolve().parents[1]
figure_dir = root / "paper" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(figure_dir / "fig1_edge_spectra.pdf")
fig.savefig(figure_dir / "fig1_edge_spectra.png", dpi=300)
print(f"Saved figure to {figure_dir / 'fig1_edge_spectra.pdf'}")
