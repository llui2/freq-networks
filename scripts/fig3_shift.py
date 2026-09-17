"""Figure 3: direct and resolvent predictions of finite edge deletion."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


red = "#D62728"
blue = "#1F77B4"

plt.rc("font", family="Times", size=11)
plt.rc("mathtext", fontset="cm")


CLIQUE_SIZE = 8
ETA_FRACTION = 0.035
POINTS = 600


def smoothed_count(laplacian, omega, eta):
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return np.array(
        [
            np.sum(
                0.5
                + np.arctan((value - eigenvalues) / eta)
                / np.pi
            )
            for value in omega
        ]
    )


graph = nx.barbell_graph(CLIQUE_SIZE, 0)
nodes = list(graph.nodes())
index = {node: i for i, node in enumerate(nodes)}

adjacency = nx.to_numpy_array(graph, nodelist=nodes)
laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

lambda_max = eigenvalues[-1]
eta = ETA_FRACTION * lambda_max
omega = np.linspace(0.0, lambda_max, POINTS)
count = smoothed_count(laplacian, omega, eta)

bridge = (CLIQUE_SIZE - 1, CLIQUE_SIZE)
internal = (0, 1)


def deletion_shift(edge):
    u, v = edge
    incidence = np.zeros(len(nodes))
    incidence[index[u]] = 1.0
    incidence[index[v]] = -1.0

    deleted = laplacian - np.outer(incidence, incidence)
    direct = smoothed_count(deleted, omega, eta) - count

    coefficients = eigenvectors.T @ incidence
    z = omega + 1j * eta
    green = np.sum(
        coefficients[:, None] ** 2
        / (eigenvalues[:, None] - z[None, :]),
        axis=0,
    )
    exact = -np.angle(1.0 - green) / np.pi

    return direct, exact


bridge_direct, bridge_exact = deletion_shift(bridge)
internal_direct, internal_exact = deletion_shift(internal)


fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.15))

x = omega / lambda_max
sample = slice(None, None, 24)

ax.plot(
    x,
    bridge_exact,
    color=red,
    linewidth=1.8,
    label="bridge edge",
)
ax.plot(
    x[sample],
    bridge_direct[sample],
    linestyle="none",
    marker="o",
    markersize=3.2,
    markerfacecolor="white",
    markeredgecolor=red,
    markeredgewidth=0.8,
)

ax.plot(
    x,
    internal_exact,
    color=blue,
    linewidth=1.8,
    label="within-clique edge",
)
ax.plot(
    x[sample],
    internal_direct[sample],
    linestyle="none",
    marker="s",
    markersize=3.0,
    markerfacecolor="white",
    markeredgecolor=blue,
    markeredgewidth=0.8,
)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.01, 0.84)
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel(r"$\Omega/\lambda_{\max}$")
ax.set_ylabel(r"$\Delta_e N_{\Omega,\eta}$")
ax.tick_params(direction="in", top=True, right=True)
ax.legend(
    loc="upper left",
    fontsize=8.7,
    frameon=False,
)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

fig.subplots_adjust(
    left=0.17,
    right=0.97,
    bottom=0.17,
    top=0.95,
)

root = Path(__file__).resolve().parents[1]
figure_dir = root / "paper" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(figure_dir / "fig3_shift.pdf")
fig.savefig(figure_dir / "fig3_shift.png", dpi=300)
print(f"Saved figure to {figure_dir / 'fig3_shift.pdf'}")
