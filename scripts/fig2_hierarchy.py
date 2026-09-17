"""Figure 2: spectral enrichment of edge classes in a hierarchical SBM."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


blue = "#1F77B4"
orange = "#FF7F0E"
green = "#2CA02C"
grey = "#C4C4C4"
dark = "#222222"

plt.rc("font", family="Times", size=11)
plt.rc("mathtext", fontset="cm")


GROUP_SIZE = 20
P_WITHIN = 0.45
P_MIDDLE = 0.10
P_OUTER = 0.025
REALIZATIONS = 30
ETA_FRACTION = 0.035
POINTS = 320
SEED = 7


def edge_class(block_u, block_v):
    if block_u == block_v:
        return "within group"
    if block_u // 2 == block_v // 2:
        return "between groups"
    return "between supergroups"


def sample_graph(seed):
    sizes = [GROUP_SIZE] * 4
    probabilities = np.full((4, 4), P_OUTER)
    np.fill_diagonal(probabilities, P_WITHIN)
    probabilities[0, 1] = probabilities[1, 0] = P_MIDDLE
    probabilities[2, 3] = probabilities[3, 2] = P_MIDDLE

    graph = nx.stochastic_block_model(
        sizes,
        probabilities,
        seed=seed,
    )

    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        for left, right in zip(components[:-1], components[1:]):
            graph.add_edge(next(iter(left)), next(iter(right)))

    block = {
        node: graph.nodes[node]["block"]
        for node in graph.nodes()
    }
    return graph, block


def class_enrichment(graph, block, omega_scaled):
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    lambda_max = eigenvalues[-1]
    eta = ETA_FRACTION * lambda_max
    omega = omega_scaled * lambda_max

    class_profiles = {
        "within group": [],
        "between groups": [],
        "between supergroups": [],
    }
    all_profiles = []

    for u, v in graph.edges():
        incidence = np.zeros(len(nodes))
        incidence[index[u]] = 1.0
        incidence[index[v]] = -1.0
        coefficients = eigenvectors.T @ incidence

        rho = (
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

        kind = edge_class(block[u], block[v])
        class_profiles[kind].append(rho)
        all_profiles.append(rho)

    mean_all = np.mean(all_profiles, axis=0)

    return {
        kind: np.mean(profiles, axis=0) / mean_all
        for kind, profiles in class_profiles.items()
    }


omega_scaled = np.linspace(0.0, 1.0, POINTS)
enrichment = {
    "within group": [],
    "between groups": [],
    "between supergroups": [],
}

for seed in range(REALIZATIONS):
    graph, block = sample_graph(seed)
    profiles = class_enrichment(graph, block, omega_scaled)

    for kind in enrichment:
        enrichment[kind].append(profiles[kind])


# A smaller realization is used only to make the three edge classes explicit.
# The ensemble statistics below use the parameters defined above.
display_sizes = [10] * 4
display_probabilities = np.full((4, 4), 0.008)
np.fill_diagonal(display_probabilities, 0.50)
display_probabilities[0, 1] = display_probabilities[1, 0] = 0.07
display_probabilities[2, 3] = display_probabilities[3, 2] = 0.07
graph = nx.stochastic_block_model(
    display_sizes,
    display_probabilities,
    seed=SEED,
)
block = {
    node: graph.nodes[node]["block"]
    for node in graph.nodes()
}

# Add one connection at each hierarchical level if a random realization misses it.
for left, right in ((0, 1), (2, 3), (1, 2)):
    left_nodes = [node for node in graph if block[node] == left]
    right_nodes = [node for node in graph if block[node] == right]
    if not any(
        graph.has_edge(u, v)
        for u in left_nodes
        for v in right_nodes
    ):
        graph.add_edge(left_nodes[0], right_nodes[0])

fig, (ax_a, ax_b) = plt.subplots(
    1,
    2,
    figsize=(6.6, 2.75),
    gridspec_kw={"width_ratios": [0.90, 1.35]},
)


# ---------------------------------------------------------------------------
# A. Hierarchical network
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
position = {}
centers = {
    0: (-0.95, 0.55),
    1: (-0.95, -0.55),
    2: (0.95, 0.55),
    3: (0.95, -0.55),
}
for node in graph.nodes():
    center = np.asarray(centers[block[node]])
    position[node] = center + 0.23 * rng.normal(size=2)

within_edges = []
middle_edges = []
outer_edges = []

for u, v in graph.edges():
    kind = edge_class(block[u], block[v])
    if kind == "within group":
        within_edges.append((u, v))
    elif kind == "between groups":
        middle_edges.append((u, v))
    else:
        outer_edges.append((u, v))

nx.draw_networkx_edges(
    graph,
    position,
    edgelist=within_edges,
    edge_color=grey,
    width=0.35,
    alpha=0.45,
    ax=ax_a,
)
nx.draw_networkx_edges(
    graph,
    position,
    edgelist=middle_edges,
    edge_color=orange,
    width=0.75,
    alpha=0.75,
    ax=ax_a,
)
nx.draw_networkx_edges(
    graph,
    position,
    edgelist=outer_edges,
    edge_color=green,
    width=0.9,
    alpha=0.85,
    ax=ax_a,
)
nx.draw_networkx_nodes(
    graph,
    position,
    node_size=12,
    node_color=dark,
    linewidths=0,
    ax=ax_a,
)

ax_a.set_axis_off()


# ---------------------------------------------------------------------------
# B. Ensemble edge-class enrichment
# ---------------------------------------------------------------------------

styles = {
    "within group": (blue, "within group"),
    "between groups": (orange, "between groups"),
    "between supergroups": (green, "between supergroups"),
}

for kind, (color, label) in styles.items():
    values = np.asarray(enrichment[kind])
    mean = values.mean(axis=0)
    std = values.std(axis=0)

    ax_b.plot(
        omega_scaled,
        mean,
        color=color,
        linewidth=1.7,
        label=label,
    )
    ax_b.fill_between(
        omega_scaled,
        np.maximum(0.0, mean - std),
        mean + std,
        color=color,
        alpha=0.13,
        linewidth=0,
    )

ax_b.axhline(
    1.0,
    color="black",
    linestyle=(0, (4, 3)),
    linewidth=0.9,
)

ax_b.set_xlim(0.0, 1.0)
ax_b.set_ylim(0.55, 2.65)
ax_b.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax_b.set_xlabel(r"$\Omega/\lambda_{\max}$")
ax_b.set_ylabel(r"$\langle\rho_e\rangle_{\rm class}/\langle\rho_e\rangle$")
ax_b.tick_params(direction="in", top=True, right=True)
ax_b.legend(
    loc="upper right",
    fontsize=8.2,
    frameon=False,
)
for spine in ax_b.spines.values():
    spine.set_linewidth(0.8)


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
    left=0.025,
    right=0.985,
    bottom=0.21,
    top=0.94,
    wspace=0.26,
)

root = Path(__file__).resolve().parents[1]
figure_dir = root / "paper" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(figure_dir / "fig2_hierarchy.pdf")
fig.savefig(figure_dir / "fig2_hierarchy.png", dpi=300)
print(f"Saved figure to {figure_dir / 'fig2_hierarchy.pdf'}")
