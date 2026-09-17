"""Figure 2: ensemble spectral enrichment of hierarchical edge classes."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


blue = "#245A9A"
orange = "#E07A1F"
green = "#238B45"

plt.rc("font", family="serif", serif=["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"], size=10)
plt.rc("mathtext", fontset="cm")


GROUP_SIZE = 15
P_WITHIN = 0.55
P_MIDDLE = 0.08
P_OUTER = 0.015
REALIZATIONS = 50
ETA_FRACTION = 0.03
POINTS = 260
MAX_FREQUENCY = 0.55


def edge_class(block_u, block_v):
    if block_u == block_v:
        return "within group"
    if block_u // 2 == block_v // 2:
        return "between groups"
    return "between supergroups"


def sample_graph(seed):
    probabilities = np.full((4, 4), P_OUTER)
    np.fill_diagonal(probabilities, P_WITHIN)
    probabilities[0, 1] = probabilities[1, 0] = P_MIDDLE
    probabilities[2, 3] = probabilities[3, 2] = P_MIDDLE

    graph = nx.stochastic_block_model(
        [GROUP_SIZE] * 4,
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

    profiles = {
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
        profiles[edge_class(block[u], block[v])].append(rho)
        all_profiles.append(rho)

    mean_all = np.mean(all_profiles, axis=0)
    return {
        kind: np.mean(values, axis=0) / mean_all
        for kind, values in profiles.items()
    }


omega_scaled = np.linspace(0.0, MAX_FREQUENCY, POINTS)
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


fig, ax = plt.subplots(1, 1, figsize=(4.35, 3.05))
styles = {
    "within group": (blue, "within group"),
    "between groups": (orange, "between groups"),
    "between supergroups": (green, "between supergroups"),
}

for kind, (color, label) in styles.items():
    values = np.asarray(enrichment[kind])
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(REALIZATIONS)
    ax.plot(
        omega_scaled,
        mean,
        color=color,
        linewidth=1.55,
        label=label,
    )
    ax.fill_between(
        omega_scaled,
        np.maximum(0.0, mean - sem),
        mean + sem,
        color=color,
        alpha=0.16,
        linewidth=0,
    )

ax.axhline(
    1.0,
    color="black",
    linestyle=(0, (4, 3)),
    linewidth=0.8,
)
ax.set_xlim(0.0, MAX_FREQUENCY)
ax.set_ylim(0.55, 4.7)
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticks([1, 2, 3, 4])
ax.set_xlabel(r"$\Omega/\lambda_{\max}$")
ax.set_ylabel(r"$\langle\rho_e\rangle_{\rm class}/\langle\rho_e\rangle$")
ax.tick_params(direction="in", top=True, right=True)
ax.legend(
    loc="upper right",
    fontsize=8.3,
    frameon=False,
    handlelength=2.8,
)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

fig.subplots_adjust(
    left=0.16,
    right=0.97,
    bottom=0.17,
    top=0.96,
)

root = Path(__file__).resolve().parents[1]
figure_dir = root / "paper" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(
    figure_dir / "fig2_hierarchy.pdf",
    bbox_inches="tight",
    pad_inches=0.03,
)
fig.savefig(
    figure_dir / "fig2_hierarchy.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
)
print(f"Saved figure to {figure_dir / 'fig2_hierarchy.pdf'}")
