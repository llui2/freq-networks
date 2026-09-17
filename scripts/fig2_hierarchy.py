"""Figure 2: decomposition of spectral sensitivity by structural edge class."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


blue = "#245A9A"
orange = "#E07A1F"
green = "#238B45"

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
REALIZATIONS = 50
ETA_FRACTION = 0.03
POINTS = 260
MAX_FREQUENCY = 0.55

CLASSES = [
    "within group",
    "between groups",
    "between supergroups",
]
COLORS = [blue, orange, green]


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


def class_profiles(graph, block, omega_scaled):
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}

    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    lambda_max = eigenvalues[-1]
    eta = ETA_FRACTION * lambda_max
    omega = omega_scaled * lambda_max

    profiles = []
    kinds = []

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

        profiles.append(rho)
        kinds.append(edge_class(block[u], block[v]))

    profiles = np.asarray(profiles)
    total = profiles.sum(axis=0)
    mean_all = profiles.mean(axis=0)

    share = {}
    enrichment = {}

    for kind in CLASSES:
        mask = np.asarray([value == kind for value in kinds])
        share[kind] = profiles[mask].sum(axis=0) / total
        enrichment[kind] = profiles[mask].mean(axis=0) / mean_all

    return share, enrichment


omega_scaled = np.linspace(0.0, MAX_FREQUENCY, POINTS)

shares = {kind: [] for kind in CLASSES}
enrichment = {kind: [] for kind in CLASSES}

for seed in range(REALIZATIONS):
    graph, block = sample_graph(seed)
    graph_share, graph_enrichment = class_profiles(
        graph,
        block,
        omega_scaled,
    )

    for kind in CLASSES:
        shares[kind].append(graph_share[kind])
        enrichment[kind].append(graph_enrichment[kind])


share_mean = {
    kind: np.mean(shares[kind], axis=0)
    for kind in CLASSES
}
enrichment_mean = {
    kind: np.mean(enrichment[kind], axis=0)
    for kind in CLASSES
}
enrichment_sem = {
    kind: np.std(enrichment[kind], axis=0, ddof=1)
    / np.sqrt(REALIZATIONS)
    for kind in CLASSES
}

peak_frequency = {
    kind: omega_scaled[np.argmax(enrichment_mean[kind])]
    for kind in CLASSES
}


fig, (ax_a, ax_b) = plt.subplots(
    2,
    1,
    figsize=(4.75, 4.65),
    sharex=True,
    gridspec_kw={"height_ratios": [0.9, 1.2]},
)

ax_a.stackplot(
    omega_scaled,
    *[share_mean[kind] for kind in CLASSES],
    colors=COLORS,
    alpha=0.88,
    linewidth=0,
    labels=CLASSES,
)
ax_a.set_ylim(0.0, 1.0)
ax_a.set_yticks([0.0, 0.5, 1.0])
ax_a.set_ylabel(r"$Q_c(\Omega)$")
ax_a.tick_params(
    direction="in",
    top=True,
    right=True,
    labelbottom=False,
)
ax_a.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.22),
    ncol=3,
    frameon=False,
    fontsize=8.0,
    handlelength=2.0,
    columnspacing=1.0,
)

for kind, color in zip(CLASSES, COLORS):
    mean = enrichment_mean[kind]
    sem = enrichment_sem[kind]

    ax_b.plot(
        omega_scaled,
        mean,
        color=color,
        linewidth=1.55,
    )
    ax_b.fill_between(
        omega_scaled,
        np.maximum(0.0, mean - sem),
        mean + sem,
        color=color,
        alpha=0.13,
        linewidth=0,
    )

ax_b.axhline(
    1.0,
    color="black",
    linestyle=(0, (4, 3)),
    linewidth=0.8,
)

for kind, color in (
    ("between supergroups", green),
    ("between groups", orange),
    ("within group", blue),
):
    frequency = peak_frequency[kind]
    ax_a.axvline(
        frequency,
        color=color,
        linewidth=0.8,
        linestyle=(0, (2.5, 2.5)),
        alpha=0.8,
    )
    ax_b.axvline(
        frequency,
        color=color,
        linewidth=0.8,
        linestyle=(0, (2.5, 2.5)),
        alpha=0.8,
    )

ax_b.set_xlim(0.0, MAX_FREQUENCY)
ax_b.set_ylim(0.5, 4.7)
ax_b.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax_b.set_yticks([1, 2, 3, 4])
ax_b.set_xlabel(r"$\Omega/\lambda_{\max}$")
ax_b.set_ylabel(r"$R_c(\Omega)$")
ax_b.tick_params(direction="in", top=True, right=True)

for ax in (ax_a, ax_b):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

for label, ax in zip(("a", "b"), (ax_a, ax_b)):
    ax.text(
        -0.13,
        1.02,
        label,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontname="DejaVu Sans",
    )

fig.subplots_adjust(
    left=0.17,
    right=0.97,
    bottom=0.12,
    top=0.89,
    hspace=0.13,
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

print("Peak frequencies:")
for kind in ("between supergroups", "between groups", "within group"):
    print(f"  {kind}: {peak_frequency[kind]:.3f}")
print(f"Saved figure to {figure_dir / 'fig2_hierarchy.pdf'}")
