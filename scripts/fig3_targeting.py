"""Figure 3: finite spectral targeting compared with static edge rankings."""

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
ETA_FRACTION = 0.03

REALIZATIONS = 40
RANDOM_DRAWS = 25
EDGE_FRACTION = 0.05
WEAKENING = 0.25

# Peak frequencies of the three class-enrichment curves in Fig. 2.
TARGETS = [0.032, 0.123, 0.486]


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

    return graph


def graph_data(graph):
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}

    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    lambda_max = eigenvalues[-1]
    eta = ETA_FRACTION * lambda_max

    edges = list(graph.edges())
    coefficients = []
    incidences = []

    for u, v in edges:
        incidence = np.zeros(len(nodes))
        incidence[index[u]] = 1.0
        incidence[index[v]] = -1.0

        incidences.append(incidence)
        coefficients.append(eigenvectors.T @ incidence)

    return (
        laplacian,
        eigenvalues,
        lambda_max,
        eta,
        np.asarray(coefficients),
        np.asarray(incidences),
    )


def edge_scores(eigenvalues, coefficients, target, eta):
    omega = target * eigenvalues[-1]

    frequency_score = (
        eta
        / np.pi
        * np.sum(
            coefficients**2
            / (
                (eigenvalues[None, :] - omega) ** 2
                + eta**2
            ),
            axis=1,
        )
    )

    nonzero = eigenvalues > 1e-10

    resistance_score = np.sum(
        coefficients[:, nonzero] ** 2
        / eigenvalues[nonzero][None, :],
        axis=1,
    )

    biharmonic_score = np.sum(
        coefficients[:, nonzero] ** 2
        / eigenvalues[nonzero][None, :] ** 2,
        axis=1,
    )

    return (
        frequency_score,
        resistance_score,
        biharmonic_score,
    )


def smoothed_count(eigenvalues, omega, eta):
    return np.sum(
        0.5
        + np.arctan((omega - eigenvalues) / eta)
        / np.pi
    )


def perturb_eigenvalues(laplacian, incidences, selection):
    perturbation = np.sum(
        [
            np.outer(incidences[index], incidences[index])
            for index in selection
        ],
        axis=0,
    )

    perturbed = laplacian - WEAKENING * perturbation
    return np.linalg.eigvalsh(perturbed)


def finite_shift(
    original_eigenvalues,
    perturbed_eigenvalues,
    omega,
    eta,
):
    return (
        smoothed_count(perturbed_eigenvalues, omega, eta)
        - smoothed_count(original_eigenvalues, omega, eta)
    )


ratios = {
    "frequency resolved": {target: [] for target in TARGETS},
    "effective resistance": {target: [] for target in TARGETS},
    "biharmonic": {target: [] for target in TARGETS},
}


for realization in range(REALIZATIONS):
    graph = sample_graph(100 + realization)

    (
        laplacian,
        eigenvalues,
        lambda_max,
        eta,
        coefficients,
        incidences,
    ) = graph_data(graph)

    number = max(
        1,
        int(round(EDGE_FRACTION * len(incidences))),
    )

    # Static rankings are target independent.
    _, resistance_score, biharmonic_score = edge_scores(
        eigenvalues,
        coefficients,
        TARGETS[0],
        eta,
    )
    resistance_selection = np.argpartition(
        resistance_score,
        -number,
    )[-number:]
    biharmonic_selection = np.argpartition(
        biharmonic_score,
        -number,
    )[-number:]

    resistance_eigenvalues = perturb_eigenvalues(
        laplacian,
        incidences,
        resistance_selection,
    )
    biharmonic_eigenvalues = perturb_eigenvalues(
        laplacian,
        incidences,
        biharmonic_selection,
    )

    frequency_eigenvalues = {}
    for target in TARGETS:
        frequency_score, _, _ = edge_scores(
            eigenvalues,
            coefficients,
            target,
            eta,
        )
        selection = np.argpartition(
            frequency_score,
            -number,
        )[-number:]

        frequency_eigenvalues[target] = perturb_eigenvalues(
            laplacian,
            incidences,
            selection,
        )

    rng = np.random.default_rng(5000 + realization)
    random_eigenvalues = []
    for _ in range(RANDOM_DRAWS):
        selection = rng.choice(
            len(incidences),
            number,
            replace=False,
        )
        random_eigenvalues.append(
            perturb_eigenvalues(
                laplacian,
                incidences,
                selection,
            )
        )

    for target in TARGETS:
        omega = target * lambda_max

        random_shift = np.mean(
            [
                finite_shift(
                    eigenvalues,
                    values,
                    omega,
                    eta,
                )
                for values in random_eigenvalues
            ]
        )

        shifts = {
            "frequency resolved": finite_shift(
                eigenvalues,
                frequency_eigenvalues[target],
                omega,
                eta,
            ),
            "effective resistance": finite_shift(
                eigenvalues,
                resistance_eigenvalues,
                omega,
                eta,
            ),
            "biharmonic": finite_shift(
                eigenvalues,
                biharmonic_eigenvalues,
                omega,
                eta,
            ),
        }

        for method, value in shifts.items():
            ratios[method][target].append(
                value / random_shift
            )


fig, ax = plt.subplots(1, 1, figsize=(4.55, 3.05))

methods = [
    "frequency resolved",
    "effective resistance",
    "biharmonic",
]
colors = [blue, orange, green]

x = np.arange(len(TARGETS))
width = 0.22

for method_index, (method, color) in enumerate(
    zip(methods, colors)
):
    means = [
        np.mean(ratios[method][target])
        for target in TARGETS
    ]
    sems = [
        np.std(
            ratios[method][target],
            ddof=1,
        )
        / np.sqrt(REALIZATIONS)
        for target in TARGETS
    ]

    ax.bar(
        x + (method_index - 1) * width,
        means,
        width,
        color=color,
        alpha=0.90,
        label=method,
        yerr=sems,
        error_kw={
            "elinewidth": 0.8,
            "capsize": 2.2,
            "capthick": 0.8,
        },
    )

ax.axhline(
    1.0,
    color="black",
    linestyle=(0, (4, 3)),
    linewidth=0.9,
    label="random",
)

ax.set_xticks(x)
ax.set_xticklabels(
    [rf"${target:.3f}$" for target in TARGETS]
)
ax.set_xlabel(r"target $\Omega_0/\lambda_{\max}$")
ax.set_ylabel("finite shift / random")
ax.set_ylim(0.0, 5.0)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.tick_params(direction="in", top=True, right=True)
ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=8.1,
)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

fig.subplots_adjust(
    left=0.16,
    right=0.97,
    bottom=0.18,
    top=0.96,
)

root = Path(__file__).resolve().parents[1]
figure_dir = root / "paper" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    figure_dir / "fig3_targeting.pdf",
    bbox_inches="tight",
    pad_inches=0.03,
)
fig.savefig(
    figure_dir / "fig3_targeting.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
)

print("Mean finite-shift gain over random:")
for target in TARGETS:
    print(f"  target {target:.3f}")
    for method in methods:
        mean = np.mean(ratios[method][target])
        print(f"    {method}: {mean:.3f}")

print(f"Saved figure to {figure_dir / 'fig3_targeting.pdf'}")
