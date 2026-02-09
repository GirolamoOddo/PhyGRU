# ============================================================
# GLOBAL IMPORTS & STYLE
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.labelsize":  18,
    "axes.titlesize":  18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

sns.set_palette("colorblind")

DT = 0.01
DATASETS = ["Sys_1", "Sys_2", "Sys_3"]

# ============================================================
# CONSISTENT COLOR MAP (used everywhere)
# ============================================================
COLORS = {
    "GT": "#000000",         # black
    "PhyGRU": "#FF7F0E",     # orange (matplotlib default orange)
    "GRU": "#1f77b4",        # blue (matplotlib default blue)
    "GRU_obs": "#2ca02c",    # green
    "InitialGuess": "#7f7f7f" # gray for initial guess in identified plot
}

# ============================================================
# UTILS
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def apply_smart_ylim(ax, arrays, force_abs=4):
    if len(arrays) == 0:
        return
    y = np.concatenate([a.ravel() for a in arrays])
    ymin, ymax = int(np.floor(y.min())), int(np.ceil(y.max()))

    if abs(ymin) > force_abs or abs(ymax) > force_abs:
        ax.set_ylim([-force_abs, force_abs])
    else:
        ax.set_ylim([ymin, ymax])

# ============================================================
# DATASET OVERVIEW (TI / TV)
# ============================================================
def plot_dataset_overview(folder, tag):
    #fig_dir = os.path.join(folder, "figures")
    fig_dir = "./figures"
    ensure_dir(fig_dir)

    fig, axes = plt.subplots(
        2 * len(DATASETS), 1,
        figsize=(13, 4.2 * len(DATASETS)),
        sharex=True
    )

    row = 0
    for name in DATASETS:
        data = np.load(f"{folder}/results_npz/{name}_dataset.npz")

        t = data["time"]
        x_tr, x_va, x_te = data["x_train"], data["x_val"], data["x_test"]
        u_tr, u_va, u_te = data["u_train"], data["u_val"], data["u_test"]

        axx = axes[row]
        axx.plot(t, x_tr, label="Train")
        axx.plot(t, x_va, label="Val")
        axx.plot(t, x_te, label="Test")
        axx.set_ylabel(r"$x(t)$")
        axx.set_title(rf"\textbf{{{name}}} — State")
        axx.grid(True, alpha=0.3)

        if row == 0:
            axx.legend()

        axu = axes[row + 1]
        axu.plot(t, u_tr)
        axu.plot(t, u_va)
        axu.plot(t, u_te)
        axu.set_ylabel(r"$u(t)$")
        axu.set_title(rf"\textbf{{{name}}} — Input")
        axu.grid(True, alpha=0.3)

        row += 2

    axes[-1].set_xlabel(r"$t\,[\mathrm{s}]$")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{tag}_dataset_overview.pdf"), bbox_inches="tight")
    plt.show()

# ============================================================
# PREDICTIONS (TI / TV / TV_STACK)
# ============================================================
def plot_predictions(folder, split, tag):
    #fig_dir = os.path.join(folder, "figures")
    fig_dir = "./figures"
    ensure_dir(fig_dir)

    fig, axes = plt.subplots(
        len(DATASETS), 1,
        figsize=(13, 4.2 * len(DATASETS)),
        sharex=True
    )

    for ax, name in zip(axes, DATASETS):
        data = np.load(f"{folder}/results_npz/{name}_predictions.npz")

        gt = data["GRTH"] if split == "test" else data["GRTH_VAL"]
        time = np.arange(len(gt)) * DT

        # Collect keys by model type to control colors and alpha
        gru_keys = [k for k in data.files if k.startswith("GRU_") and not k.startswith("GRU_obs") and ( (split=="test" and not k.endswith("_val")) or (split=="val" and k.endswith("_val")) )]
        gru_obs_keys = [k for k in data.files if k.startswith("GRU_obs") and ( (split=="test" and not k.endswith("_val")) or (split=="val" and k.endswith("_val")) )]
        phy_keys = [k for k in data.files if k.startswith("PhyGRU") and ( (split=="test" and not k.endswith("_val")) or (split=="val" and k.endswith("_val")) )]

        curves = [gt]

        # GRU_obs (green)
        n_obs = len(gru_obs_keys)
        if n_obs > 0:
            alpha_member = 0.4 if n_obs > 1 else 0.6
            for k in gru_obs_keys:
                ax.plot(time, data[k], color=COLORS["GRU_obs"], alpha=alpha_member, label="_nolegend_")
                curves.append(data[k])
            # mean
            mean_obs = np.mean([data[k] for k in gru_obs_keys], axis=0)
            ax.plot(time, mean_obs, color=COLORS["GRU_obs"], lw=2, alpha=1.0, label=r"$\mathrm{GRU_{obs}}$")
            curves.append(mean_obs)

        # GRU (blue)
        n_gru = len(gru_keys)
        if n_gru > 0:
            alpha_member = 0.4 if n_gru > 1 else 0.6
            for k in gru_keys:
                ax.plot(time, data[k], color=COLORS["GRU"], alpha=alpha_member, label="_nolegend_")
                curves.append(data[k])
            mean_gru = np.mean([data[k] for k in gru_keys], axis=0)
            ax.plot(time, mean_gru, color=COLORS["GRU"], lw=2, alpha=1.0, label=r"$\mathrm{GRU}$")
            curves.append(mean_gru)

        # PhyGRU (orange)
        n_phy = len(phy_keys)
        if n_phy > 0:
            alpha_member = 0.4 if n_phy > 1 else 0.6
            for k in phy_keys:
                ax.plot(time, data[k], color=COLORS["PhyGRU"], alpha=alpha_member, label="_nolegend_")
                curves.append(data[k])
            mean_phy = np.mean([data[k] for k in phy_keys], axis=0)
            ax.plot(time, mean_phy, color=COLORS["PhyGRU"], lw=2, alpha=1.0, label=r"$\mathrm{PhyGRU}$")
            curves.append(mean_phy)

        # Ground truth (black dashed)
        ax.plot(time, gt, color=COLORS["GT"], linestyle="--", lw=2, label=r"Ground Truth")
        apply_smart_ylim(ax, curves)

        ax.set_title(rf"\textbf{{{name}}} — {split.upper()}")
        ax.set_ylabel(r"$x(t)$")
        ax.grid(True, alpha=0.3)

        if ax is axes[0]:
            ax.legend()

    axes[-1].set_xlabel(r"$t\,[\mathrm{s}]$")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{tag}_{split}.pdf"), bbox_inches="tight")
    plt.show()

# ============================================================
# UPDATE GATE DIAGNOSTICS (ALL LATENTS)
# ============================================================
def run_update_gate_analysis_TI(
    dataset_name="Sys_3",
    latent_dim=2,
    T=6000,
    dt=0.01,
    base_dir="TI",
    save_fig=True
):
    """
    Update-gate analysis for PhyGRU using TI checkpoints.
    Computes physics candidate, all latent candidates, and gate activation.
    Fully self-contained and ready to paste.
    """

    import os
    import math
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    torch.set_grad_enabled(False)

    # ------------------ COLORS ------------------
    COLORS = {
        "GT": "black",
        "PhyGRU": "#1f77b4",  # blue
        "GRU": "#ff7f0e",     # orange
        "GRU_obs": "#2ca02c", # green (will cycle for multiple latents)
    }

    # ------------------ PATHS ------------------
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    #fig_dir  = os.path.join(base_dir, "figures")
    fig_dir = "./figures"
    os.makedirs(fig_dir, exist_ok=True)

    ckpt_path = os.path.join(
        ckpt_dir, f"{dataset_name}_PhyGRU_l{latent_dim}_best.pt"
    )

    time = np.arange(T) * dt

    # ============================================================
    # DATASET GENERATORS
    # ============================================================
    def generate_data_1(u_fn):
        x, xd = 0.0, 0.0
        xs, us = [], []
        for t_ in range(T):
            u = u_fn(t_)
            xdd = (u - 0.5 * xd - 0.2 * x)
            xd += dt * xdd
            x  += dt * xd
            xs.append([x])
            us.append([u])
        return torch.tensor(xs), torch.tensor(us)

    def generate_data_3(u_fn):
        x, xd, xdd = 0.0, 0.0, 0.0
        xs, us = [], []
        for t in range(T):
            u = u_fn(t)
            xddd = u - 3.0 * xdd - 2.0 * math.tanh(xd) - 0.1 * math.tanh(x)
            xdd += dt * xddd
            xd  += dt * xdd
            x   += dt * xd
            xs.append([x])
            us.append([u])
        return torch.tensor(xs), torch.tensor(us)

    if dataset_name == "Sys_1":
        x_test, u_test = generate_data_1(lambda t: math.tanh(
            (0.5 - 0.00005 * t) * (
                (0.30 - 0.002 * t) * math.sin((0.00050 + 0.0000005 * t) * t)
                + (0.15 + 0.001 * t) * math.sin((0.000004 - 0.000001 * t) * t)
            )
        ))
    else:  # Sys_3
        x_test, u_test = generate_data_3(lambda t: math.tanh(
            (0.5 - 0.00005 * t) * (
                (0.30 - 0.002 * t) * math.sin((0.00050 + 0.0000005 * t) * t)
                + (0.15 + 0.001 * t) * math.sin((0.000004 - 0.000001 * t) * t)
            )
        ))

    x_test = x_test.unsqueeze(0)
    u_test = u_test.unsqueeze(0)
    x_test /= torch.max(torch.abs(x_test))
    u_test /= torch.max(torch.abs(u_test))

    # ============================================================
    # PHYSICS LAW
    # ============================================================
    class MassSpringDamperLaw(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(0.5))
            self.b = nn.Parameter(torch.tensor(0.6))
            self.c = nn.Parameter(torch.tensor(0.7))

        def forward(self, state, u):
            x, xd = state[:, 0], state[:, 1]
            u = u.squeeze()
            xdd = (u - self.b * xd - self.c * x) / (self.a + 1e-12)
            return torch.stack([xd, xdd], dim=1)

    # ============================================================
    # PHYGRU
    # ============================================================
    class PhyGRUCell(nn.Module):
        def __init__(self, state_dim, input_dim, physics_law, latent_dim):
            super().__init__()
            total = state_dim + latent_dim
            self.state_dim = state_dim
            self.latent_dim = latent_dim
            self.physics_law = physics_law
            self.latent_dyn = nn.Linear(total + input_dim, latent_dim)
            self.z_gate = nn.Sequential(nn.Linear(total + input_dim, total), nn.Sigmoid())

        def forward(self, state, u):
            phys_dot  = self.physics_law(state[:, :self.state_dim], u)
            phys_next = state[:, :self.state_dim] + dt * phys_dot
            latent = state[:, self.state_dim:]
            latent_dot = self.latent_dyn(torch.cat([state, u], dim=1))
            latent_next = latent + dt * latent_dot
            candidate = torch.cat([phys_next, latent_next], dim=1)
            z = self.z_gate(torch.cat([state, u], dim=1))
            state_next = z * candidate + (1.0 - z) * state
            return state_next, phys_next, latent_next, candidate, z

    class PhyGRU(nn.Module):
        def __init__(self, physics_law, state_dim, input_dim, latent_dim):
            super().__init__()
            self.cell = PhyGRUCell(state_dim, input_dim, physics_law, latent_dim)

    # ============================================================
    # LOAD MODEL
    # ============================================================
    model = PhyGRU(MassSpringDamperLaw(), state_dim=2, input_dim=1, latent_dim=latent_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # ============================================================
    # INFERENCE + LOGGING
    # ============================================================
    state = torch.zeros(1, 2 + latent_dim)
    y_pred = []
    candidate_phys = []
    candidate_latents = [[] for _ in range(latent_dim)]
    gate_x = []

    for t in range(T):
        u_t = u_test[:, t]
        state, phys, latent, candidate, z = model.cell(state, u_t)
        y_pred.append(state[0, 0].item())
        candidate_phys.append(candidate[0, 0].item())
        # log all latent candidates
        for i in range(latent_dim):
            candidate_latents[i].append(candidate[0, 2 + i].item())
        gate_x.append(z[0, 0].item())

    y_pred = np.array(y_pred)
    candidate_phys = np.array(candidate_phys)
    candidate_latents = [np.array(c) for c in candidate_latents]
    gate_x = np.array(gate_x)

    # ============================================================
    # PLOTS (GT vs pred, physics + all latents, gate)
    # ============================================================
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # Ground truth vs prediction
    axes[0].plot(time, x_test[0, :, 0], linestyle="--", color=COLORS["GT"], lw=2, label="Ground Truth")
    axes[0].plot(time, y_pred, color=COLORS["PhyGRU"], lw=2, label="PhyGRU")
    axes[0].set_ylabel(r"$x(t)$")
    #axes[0].set_title(rf"\textbf{{{dataset_name}}} --- Prediction")
    axes[0].set_title(rf"\textbf{{{dataset_name}}}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Candidates: physics + all latent candidates
    axes[1].plot(time, candidate_phys, label="Physics Candidate", color=COLORS["PhyGRU"])
    for i, cl in enumerate(candidate_latents):
        axes[1].plot(time, cl, label=f"Latent {i+1} Candidate", color=plt.cm.Set2(i % 8))
    axes[1].set_ylabel(r"$\tilde{\mathbf{h}}$")
    #axes[1].set_title(rf"\textbf{{{dataset_name}}} --- Candidates (physics + latents)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gate
    axes[2].plot(time, gate_x, color=COLORS["GRU"], label=r"Gate $z_x$")
    axes[2].set_ylabel(r"$z(t)$")
    axes[2].set_xlabel(r"$t\,[\mathrm{s}]$")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    #axes[2].legend()

    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(fig_dir, f"TI_update_gate_{dataset_name}.pdf"), bbox_inches="tight")
    plt.show()

    return {
        "time": time,
        "y_pred": y_pred,
        "candidate_phys": candidate_phys,
        "candidate_latents": candidate_latents,
        "gate_x": gate_x,
    }

# ============================================================
# IDENTIFIED MODELS PLOT (CONSISTENT STYLE)
# ============================================================
def plot_identified_phygru_models(
    dataset_name="Sys_1",
    results_dir="results_npz",
    dt=0.01,
    save_fig=True,
    fig_dir="figures"
):
    """
    Plot comparison between:
      - Ground Truth trajectory
      - Initial Guess physics
      - Identified PhyGRU physics parameters (latent models)
    LaTeX-styled, publication-ready and consistent with other figures.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ------------------ STYLE ------------------
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    sns.set_palette("colorblind")
    #sns.set_style("whitegrid")

    os.makedirs(fig_dir, exist_ok=True)

    # ------------------ PARAMETERS ------------------
    GT = [1.0, 0.5, 0.2]   # Ground Truth (a, b, c)
    IG = [0.5, 0.6, 0.7]   # Initial Guess

    identified = {
        "l0": [0.631, 0.915, 0.826],
        "l1": [0.382, 0.735, 0.800],
        "l2": [0.658, 0.698, 0.841],
        "l3": [0.436, 0.832, 0.858],
    }

    # ------------------ MSD SIM ------------------
    def msd_simulate(a, b, c, u_seq):
        x, xd = 0.0, 0.0
        xs = []
        for u in u_seq:
            xdd = (u - b * xd - c * x) / (a + 1e-12)
            xd += dt * xdd
            x  += dt * xd
            xs.append(x)
        return np.array(xs)

    # ------------------ LOAD DATA ------------------
    data = np.load(os.path.join(results_dir, f"{dataset_name}_predictions.npz"))
    u_test = data["U"]
    gt_test = data["GRTH"]

    time = np.arange(len(gt_test)) * dt

    # Normalize input (training-consistent)
    u_abs_max = np.max(np.abs(u_test))
    if u_abs_max == 0:
        u_abs_max = 1.0
    u_test_norm = u_test / u_abs_max

    # ------------------ SIMULATIONS ------------------
    sim_gt = msd_simulate(*GT, u_test_norm)
    sim_ig = msd_simulate(*IG, u_test_norm)
    sim_ids = {k: msd_simulate(*v, u_test_norm) for k, v in identified.items()}

    # ------------------ PLOT ------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    # Ground Truth (black dashed)
    ax.plot(time, gt_test, "k--", lw=2.5, label=r"Ground Truth")

    # Initial Guess (gray)
    COLORS = {"InitialGuess": "#7f7f7f"}  # keep consistent with other figures
    ax.plot(time, sim_ig, "--", lw=2.5, color=COLORS["InitialGuess"], label=r"Initial Guess")

    # Identified latent models (Set2 palette)
    colors = sns.color_palette("Set2", len(sim_ids))
    for i, (k, xsim) in enumerate(sim_ids.items()):
        ax.plot(time, xsim, lw=2.0, color=colors[i], label=rf"Identified {k}")

    ax.set_xlabel(r"$t\,[\mathrm{s}]$")
    ax.set_ylabel(r"$x(t)$")
    #ax.set_title(r"\textbf{{{dataset_name}}} --- Identified PhyGRU Physics Parameters")
    #ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(fig_dir, f"{dataset_name}_identified_phygru.pdf"), bbox_inches="tight")

    plt.show()


def plot_TI_TV_datasets_overview(
    T=6000,
    dt=0.01,
    mode="TI",          # "TI" or "TV"
    save_fig=True,
    fig_dir="figures"
):
    """
    Plot normalized Train / Validation / Test trajectories
    for TI or TV systems (Sys_1, Sys_2, Sys_3) in LaTeX style.
    """

    import os
    import math
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_palette("colorblind")

    os.makedirs(fig_dir, exist_ok=True)

    t = np.arange(T) * dt

    # ============================================================
    # DATASET GENERATORS
    # ============================================================
    if mode == "TI":

        def generate_data_1(u_fn):
            x, xd = 0.0, 0.0
            xs, us = [], []
            for t_ in range(T):
                u = u_fn(t_)
                xdd = (u - 0.5 * xd - 0.2 * x)
                xd += dt * xdd
                x  += dt * xd
                xs.append([x])
                us.append([u])
            return torch.tensor(xs), torch.tensor(us)

        def generate_data_2(u_fn):
            x, xd = 0.0, 0.0
            xs, us = [], []
            for t_ in range(T):
                u = u_fn(t_)
                xdd = (u - 0.5 * xd - 0.2 * x + math.tanh(x * xd))
                xd += dt * xdd
                x  += dt * xd
                xs.append([x])
                us.append([u])
            return torch.tensor(xs), torch.tensor(us)

        def generate_data_3(u_fn):
            x, xd, xdd = 0.0, 0.0, 0.0
            xs, us = [], []
            for t_ in range(T):
                u = u_fn(t_)
                xddd = u - 3 * xdd - 2 * math.tanh(xd) - 0.1 * math.tanh(x)
                xdd += dt * xddd
                xd  += dt * xdd
                x   += dt * xd
                xs.append([x])
                us.append([u])
            return torch.tensor(xs), torch.tensor(us)

    elif mode == "TV":

        def generate_data_1(u_fn):
            x, xd = 0.0, 0.0
            xs, us = [], []
            for t_ in range(T):
                u = u_fn(t_)
                xdd = (u - (0.5 + 0.00009 * t_) * xd - (0.2 + 0.0000001 * t_**2) * x)
                xd += dt * xdd
                x  += dt * xd
                xs.append([x])
                us.append([u])
            return torch.tensor(xs), torch.tensor(us)

        def generate_data_2(u_fn):
            x, xd = 0.0, 0.0
            xs, us = [], []
            for t_ in range(T):
                u = u_fn(t_)
                xdd = (
                    u
                    - (0.5 + 0.00009 * t_) * xd
                    - (0.2 + 0.0000001 * t_**2) * x
                    + (1.0 + 0.00009 * t_) * math.tanh(x * xd)
                )
                xd += dt * xdd
                x  += dt * xd
                xs.append([x])
                us.append([u])
            return torch.tensor(xs), torch.tensor(us)

        def generate_data_3(u_fn):
            x, xd, xdd = 0.0, 0.0, 0.0
            xs, us = [], []
            for t_ in range(T):
                u = u_fn(t_)
                xddd = (
                    u
                    - (3.0 - 0.0002 * t_) * xdd
                    - (2.0 + 0.0009 * t_) * math.tanh(xd)
                    - (0.1 + 0.0000003 * t_**2) * math.tanh(x)
                )
                xdd += dt * xddd
                xd  += dt * xdd
                x   += dt * xd
                xs.append([x])
                us.append([u])
            return torch.tensor(xs), torch.tensor(us)

    else:
        raise ValueError("mode must be 'TI' or 'TV'")

    systems = [
        ("Sys_1", generate_data_1),
        ("Sys_2", generate_data_2),
        ("Sys_3", generate_data_3),
    ]

    # ============================================================
    # INPUT FUNCTIONS
    # ============================================================
    u_train_fn = lambda t: math.tanh(
        (0.3 - 0.00005 * t)
        * (
            (0.25 - 0.001 * t) * math.sin((0.00007 + 0.0000001 * t) * t)
            + (0.10 + 0.001 * t) * math.sin((0.000001 - 0.000001 * t) * t)
        )
    )

    u_val_fn = lambda t: math.tanh(
        (0.4 - 0.00005 * t)
        * (
            (0.35 - 0.003 * t) * math.sin((0.00009 + 0.0000002 * t) * t)
            + (0.15 + 0.001 * t) * math.sin((0.000003 - 0.000001 * t) * t)
        )
    )

    u_test_fn = lambda t: math.tanh(
        (0.5 - 0.00005 * t)
        * (
            (0.30 - 0.002 * t) * math.sin((0.00050 + 0.0000005 * t) * t)
            + (0.15 + 0.001 * t) * math.sin((0.000004 - 0.000001 * t) * t)
        )
    )

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(
        nrows=2 * len(systems),
        ncols=1,
        figsize=(13, 4.2 * len(systems)),
        sharex=True
    )

    row = 0
    for name, data_fn in systems:

        x_train, u_train = data_fn(u_train_fn)
        x_val,   u_val   = data_fn(u_val_fn)
        x_test,  u_test  = data_fn(u_test_fn)

        x_abs_max = max(
            float(torch.max(torch.abs(x_train))),
            float(torch.max(torch.abs(x_val))),
            float(torch.max(torch.abs(x_test)))
        ) or 1.0

        u_abs_max = max(
            float(torch.max(torch.abs(u_train))),
            float(torch.max(torch.abs(u_val))),
            float(torch.max(torch.abs(u_test)))
        ) or 1.0

        x_train /= x_abs_max
        x_val   /= x_abs_max
        x_test  /= x_abs_max

        u_train /= u_abs_max
        u_val   /= u_abs_max
        u_test  /= u_abs_max

        ax_x = axes[row]
        ax_x.plot(t, x_train.squeeze(), label=r"Train")
        ax_x.plot(t, x_val.squeeze(),   label=r"Validation")
        ax_x.plot(t, x_test.squeeze(),  label=r"Test")
        ax_x.set_ylabel(r"$x(t)$")
        ax_x.set_title(rf"\textbf{{{name}}} --- State")
        ax_x.grid(True, alpha=0.3)

        if row == 0:
            ax_x.legend(loc="upper right")

        ax_u = axes[row + 1]
        ax_u.plot(t, u_train.squeeze())
        ax_u.plot(t, u_val.squeeze())
        ax_u.plot(t, u_test.squeeze())
        ax_u.set_ylabel(r"$u(t)$")
        ax_u.set_title(rf"\textbf{{{name}}} --- Input")
        ax_u.grid(True, alpha=0.3)

        row += 2

    axes[-1].set_xlabel(r"$t\,[\mathrm{s}]$")
    plt.tight_layout()

    if save_fig:
        plt.savefig(
            os.path.join(fig_dir, f"{mode}_datasets_overview.pdf"),
            bbox_inches="tight"
        )

    plt.show()



def compare_incremental_stability_phygru_vs_gru(
    dataset_name="Sys_1",
    latent_dim=1,
    T=6000,
    dt=0.01,
    N_PERTURB=100,
    perturb_range=3.0,
    base_dir="TI",
    fig_dir="./figures",
    save_fig=True,
    device="cpu",
    gru_hidden=32,
    colors=None
):
    """
    Compare incremental stability of a trained PhyGRU vs a standard GRU (hidden=gru_hidden).
    - Runs N_PERTURB perturbed initial conditions for each model (same input sequence).
    - Plots: (top) all perturbed trajectories + references + GT; (bottom) distances to reference (log-scale) and mean.
    - Attempts to load GRU checkpoint flexibly; if incompatible, GRU comparison is skipped.
    - Returns a dict with arrays for later inspection.
    """
    import os
    import math
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    torch.set_grad_enabled(False)
    dev = torch.device(device)

    if colors is None:
        # fallback to global COLORS if present in caller's namespace
        try:
            colors = globals()["COLORS"]
        except Exception:
            colors = {
                "GT": "#000000",
                "PhyGRU": "#FF7F0E",
                "GRU": "#1f77b4",
                "GRU_obs": "#2ca02c",
                "InitialGuess": "#7f7f7f"
            }

    os.makedirs(fig_dir, exist_ok=True)

    # ------------------- input / GT generators (Sys_1 used as GT) -------------------
    def generate_input_seq(T):
        u = []
        for t in range(T):
            val = math.tanh(
                (0.5 - 0.00005 * t) * (
                    (0.30 - 0.002 * t) * math.sin((0.00050 + 0.0000005 * t) * t)
                    + (0.15 + 0.001 * t) * math.sin((0.000004 - 0.000001 * t) * t)
                )
            )
            u.append([val])
        u = torch.tensor(u, dtype=torch.float32).unsqueeze(0)  # (1,T,1)
        u /= torch.max(torch.abs(u)) + 1e-12
        return u

    def generate_sys1_trajectory(T, dt):
        x, xd = 0.0, 0.0
        xs = []
        for t in range(T):
            u = math.tanh(
                (0.5 - 0.00005 * t) * (
                    (0.30 - 0.002 * t) * math.sin((0.00050 + 0.0000005 * t) * t)
                    + (0.15 + 0.001 * t) * math.sin((0.000004 - 0.000001 * t) * t)
                )
            )
            xdd = (u - 0.5 * xd - 0.2 * x)
            xd += dt * xdd
            x  += dt * xd
            xs.append(x)
        xs = np.array(xs)
        xs /= np.max(np.abs(xs)) + 1e-12
        return xs

    u_seq = generate_input_seq(T)            # (1,T,1)
    time = np.arange(T) * dt
    gt_traj = generate_sys1_trajectory(T, dt)  # normalized GT

    # ------------------- PhyGRU model definition & loading -------------------
    class MassSpringDamperLaw(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(0.5))
            self.b = nn.Parameter(torch.tensor(0.6))
            self.c = nn.Parameter(torch.tensor(0.7))
        def forward(self, state, u):
            x, xd = state[:, 0], state[:, 1]
            u = u[:, 0]
            xdd = (u - self.b * xd - self.c * x) / (self.a + 1e-12)
            return torch.stack([xd, xdd], dim=1)

    class PhyGRUCell(nn.Module):
        def __init__(self, state_dim, input_dim, physics_law, latent_dim):
            super().__init__()
            total = state_dim + latent_dim
            self.state_dim = state_dim
            self.latent_dim = latent_dim
            self.physics_law = physics_law
            self.latent_dyn = nn.Linear(total + input_dim, latent_dim)
            self.z_gate = nn.Sequential(nn.Linear(total + input_dim, total), nn.Sigmoid())
        def forward(self, state, u):
            phys_dot = self.physics_law(state[:, :self.state_dim], u)
            phys_next = state[:, :self.state_dim] + dt * phys_dot
            latent = state[:, self.state_dim:]
            latent_dot = self.latent_dyn(torch.cat([state, u], dim=1))
            latent_next = latent + dt * latent_dot
            candidate = torch.cat([phys_next, latent_next], dim=1)
            z = self.z_gate(torch.cat([state, u], dim=1))
            state_next = z * candidate + (1.0 - z) * state
            return state_next, phys_next, latent_next, candidate, z

    class PhyGRU(nn.Module):
        def __init__(self, physics_law, state_dim, input_dim, latent_dim):
            super().__init__()
            self.cell = PhyGRUCell(state_dim, input_dim, physics_law, latent_dim)

    phyckpt = os.path.join(base_dir, "checkpoints", f"{dataset_name}_PhyGRU_l{latent_dim}_best.pt")
    if not os.path.exists(phyckpt):
        raise FileNotFoundError(f"[compare_incremental_stability] PhyGRU checkpoint not found at {phyckpt}")

    phy_model = PhyGRU(MassSpringDamperLaw(), state_dim=2, input_dim=1, latent_dim=latent_dim)
    phy_model.load_state_dict(torch.load(phyckpt, map_location=dev))
    phy_model.to(dev)
    phy_model.eval()

    # ------------------- Std GRU flexible wrapper & loading attempts -------------------
    class StdGRUModelFC(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32, n_layers=1):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
        def forward_step(self, h_prev, u_t):
            # h_prev: (n_layers, batch, hidden)
            u_t = u_t.unsqueeze(1)  # (batch,1,input_dim)
            out, h_next = self.gru(u_t, h_prev)
            y = self.fc(out[:,0,:])
            return h_next, y

    class StdGRUModelReadout(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32, n_layers=1):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
            self.readout = nn.Linear(hidden_dim, 1)
        def forward_step(self, h_prev, u_t):
            u_t = u_t.unsqueeze(1)
            out, h_next = self.gru(u_t, h_prev)
            y = self.readout(out[:,0,:])
            return h_next, y

    gru_ckpt = os.path.join(base_dir, "checkpoints", f"{dataset_name}_GRU_h{gru_hidden}_best.pt")
    gru_loaded = False
    gru_model = None
    if os.path.exists(gru_ckpt):
        sd = torch.load(gru_ckpt, map_location=dev)
        # try FC wrapper first
        try:
            temp = StdGRUModelFC(input_dim=1, hidden_dim=gru_hidden)
            temp.load_state_dict(sd)
            gru_model = temp.to(dev)
            gru_loaded = True
        except Exception:
            # try readout wrapper
            try:
                temp = StdGRUModelReadout(input_dim=1, hidden_dim=gru_hidden)
                temp.load_state_dict(sd)
                gru_model = temp.to(dev)
                gru_loaded = True
            except Exception:
                # try partial (intersection) load into FC
                try:
                    temp = StdGRUModelFC(input_dim=1, hidden_dim=gru_hidden)
                    common = {k: v for k, v in sd.items() if k in temp.state_dict().keys()}
                    if len(common) > 0:
                        temp.load_state_dict(common, strict=False)
                        gru_model = temp.to(dev)
                        gru_loaded = True
                except Exception:
                    gru_loaded = False

    if not gru_loaded:
        print(f"[compare_incremental_stability] [WARN] No compatible GRU checkpoint loaded from {gru_ckpt}. GRU comparison will be skipped.")
    else:
        gru_model.eval()
        print(f"[compare_incremental_stability] Loaded GRU model from {gru_ckpt}")

    # ------------------- simulation helpers -------------------
    def simulate_phygru(state_init, u_seq):
        state = state_init.clone().to(dev)
        traj = []
        for t in range(T):
            u_t = u_seq[:, t, :].to(dev)                   # (1,1)
            state, _, _, _, _ = phy_model.cell(state, u_t)
            traj.append(state[0, 0].item())
        traj = np.array(traj)
        # normalize to max abs (same as GT normalization)
        if np.max(np.abs(traj)) > 0:
            traj = traj / (np.max(np.abs(traj)) + 1e-12)
        return traj

    def simulate_gru(h0, u_seq):
        h = h0.clone().to(dev)
        traj = []
        for t in range(T):
            u_t = u_seq[:, t, :].to(dev)
            h, y = gru_model.forward_step(h, u_t)
            traj.append(float(y[0, 0]))
        traj = np.array(traj)
        if np.max(np.abs(traj)) > 0:
            traj = traj / (np.max(np.abs(traj)) + 1e-12)
        return traj

    # ------------------- compute references -------------------
    ref_state_phy = torch.zeros(1, 2 + latent_dim, dtype=torch.float32, device=dev)
    ref_traj_phy = simulate_phygru(ref_state_phy, u_seq)

    if gru_loaded:
        # prepare reference hidden with shape (n_layers, batch, hidden)
        if hasattr(gru_model, "n_layers"):
            n_layers = gru_model.n_layers
        else:
            n_layers = 1
        h0_ref = torch.zeros(n_layers, 1, gru_model.hidden_dim, dtype=torch.float32, device=dev)
        ref_traj_gru = simulate_gru(h0_ref, u_seq)
    else:
        ref_traj_gru = None

    # ------------------- run perturbation experiments -------------------
    all_trajs_phy = []
    all_dists_phy = []

    for i in range(N_PERTURB):
        pert = (torch.rand(1, 2 + latent_dim, dtype=torch.float32) - 0.5) * 2 * perturb_range
        traj = simulate_phygru(pert, u_seq)
        all_trajs_phy.append(traj)
        all_dists_phy.append(np.abs(traj - ref_traj_phy))

    all_trajs_phy = np.array(all_trajs_phy)
    all_dists_phy = np.array(all_dists_phy)

    all_trajs_gru = None
    all_dists_gru = None
    if gru_loaded:
        all_trajs_gru = []
        all_dists_gru = []
        for i in range(N_PERTURB):
            pert_h = (torch.rand(n_layers, 1, gru_model.hidden_dim, dtype=torch.float32) - 0.5) * 2 * perturb_range
            traj = simulate_gru(pert_h, u_seq)
            all_trajs_gru.append(traj)
            all_dists_gru.append(np.abs(traj - ref_traj_gru))
        all_trajs_gru = np.array(all_trajs_gru)
        all_dists_gru = np.array(all_dists_gru)

    # ------------------- plotting -------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Top: trajectories
    ax = axes[0]
    # PhyGRU perturbed
    for traj in all_trajs_phy:
        ax.plot(time, traj, color=colors["PhyGRU"], alpha=0.25, lw=0.8)
    ax.plot(time, ref_traj_phy, color=colors["PhyGRU"], lw=2.0, label=r"PhyGRU reference")

    # GRU perturbed + ref
    if gru_loaded:
        for traj in all_trajs_gru:
            ax.plot(time, traj, color=colors["GRU"], alpha=0.25, lw=0.8)
        ax.plot(time, ref_traj_gru, color=colors["GRU"], lw=2.0, label=r"GRU reference")

    # Ground truth
    ax.plot(time, gt_traj, "--", color=colors["GT"], lw=2.0, label=r"Ground Truth")

    ax.set_ylabel(r"$x(t)$")
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom: distances (log scale) + mean
    ax2 = axes[1]
    for dist in all_dists_phy:
        ax2.plot(time, dist, color=colors["PhyGRU"], alpha=0.12, lw=0.7)
    mean_phy = np.mean(all_dists_phy, axis=0)
    ax2.plot(time, mean_phy, color=colors["PhyGRU"], lw=2.2, label=r"PhyGRU mean dist")

    if gru_loaded:
        for dist in all_dists_gru:
            ax2.plot(time, dist, color=colors["GRU"], alpha=0.12, lw=0.7)
        mean_gru = np.mean(all_dists_gru, axis=0)
        ax2.plot(time, mean_gru, color=colors["GRU"], lw=2.2, label=r"GRU mean dist")

    ax2.set_ylim([1e-6, max(3.0, np.max(mean_phy) * 2 + 1e-12)])
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$t\,[\mathrm{s}]$")
    ax2.set_ylabel(r"$|x_{\mathrm{pert}}(t) - x_{\mathrm{ref}}(t)|$")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_pdf = os.path.join(fig_dir, f"{dataset_name}_phygru_vs_gru_incremental_stability.pdf")
    if save_fig:
        plt.savefig(out_pdf, bbox_inches="tight")
        print(f"[compare_incremental_stability] Saved figure to {out_pdf}")

    plt.show()

    return {
        "time": time,
        "gt_traj": gt_traj,
        "ref_traj_phy": ref_traj_phy,
        "all_trajs_phy": all_trajs_phy,
        "all_dists_phy": all_dists_phy,
        "ref_traj_gru": ref_traj_gru,
        "all_trajs_gru": all_trajs_gru,
        "all_dists_gru": all_dists_gru,
        "phy_model": phy_model,
        "gru_model": gru_model if gru_loaded else None,
    }


#############
# DT STUDY

def plot_dt_timeseries(
    results_dir="results_npz",
    fig_dir="figures",
    dataset_name="Sys_1",
    train_dts=(0.01, 0.9),
    test_dts=(0.005, 0.01, 0.9),
    dt_for_timeaxis=DT,
    save_fig=True
):
    """
    Plot time-series predictions from PhyGRU cross-evaluation .npz files.
    Paper-consistent styling: vertical stacked subplots, same figsize and fonts used across the project.

    Files expected (flexible):
      {results_dir}/{dataset_name}_PhyGRU_modeldt{train_dt}_results.npz
    Keys expected (flexible):
      modeldt_{test_dt}_pred, modeldt_{test_dt}_true, modeldt_{test_dt}_mse, modeldt_{test_dt}_spearman
    The finder tolerates several float string formats (e.g. '0.01', '0.010', '0.01' etc).
    """

    _ensure_dir(fig_dir)

    # local helper: candidate string forms for floats
    def _fmt_candidates(x):
        return [str(x), f"{x:.2f}", f"{x:.3f}", f"{x:g}"]

    # local helper: find npz for a given train_dt (tries multiple formats and scans folder)
    def _find_npz_for_train_dt_local(results_dir, dataset_name, tr_dt):
        candidates = _fmt_candidates(tr_dt)
        for c in candidates:
            fname = os.path.join(results_dir, f"{dataset_name}_PhyGRU_modeldt{c}_results.npz")
            if os.path.exists(fname):
                return np.load(fname, allow_pickle=True), fname
        prefix = f"{dataset_name}_PhyGRU_modeldt"
        if not os.path.isdir(results_dir):
            raise FileNotFoundError(f"Results dir '{results_dir}' not found.")
        files = [f for f in os.listdir(results_dir) if f.startswith(prefix) and f.endswith("_results.npz")]
        for f in files:
            core = f[len(prefix):-len("_results.npz")]
            try:
                val = float(core)
                if abs(val - float(tr_dt)) < 1e-9:
                    return np.load(os.path.join(results_dir, f), allow_pickle=True), os.path.join(results_dir, f)
            except Exception:
                for c in candidates:
                    if c in core:
                        return np.load(os.path.join(results_dir, f), allow_pickle=True), os.path.join(results_dir, f)
        raise FileNotFoundError(f"No results .npz found for train_dt={tr_dt} in {results_dir}")

    # local helper: find key for test_dt and suffix inside a npz file
    def _find_key_for_test_dt_local(npzfile, te_dt, suffix):
        candidates = _fmt_candidates(te_dt)
        prefixes = ["modeldt_", "td_"]
        for pref in prefixes:
            for cand in candidates:
                k = f"{pref}{cand}_{suffix}"
                if k in npzfile.files:
                    return k
        # regex fallback: try parse numeric part
        pat = re.compile(r"^(?:td_|modeldt_)(.+)_" + re.escape(suffix) + r"$")
        for k in npzfile.files:
            m = pat.match(k)
            if m:
                try:
                    val = float(m.group(1))
                    if abs(val - float(te_dt)) < 1e-9:
                        return k
                except Exception:
                    continue
        return None

    # --------------------------
    # Load requested train_dts
    # --------------------------
    loaded = {}
    for tr in train_dts:
        npz, path = _find_npz_for_train_dt_local(results_dir, dataset_name, tr)
        loaded[tr] = {"npz": npz, "path": path}

    # --------------------------
    # Determine time axis from any available '*_true' key
    # --------------------------
    sample_npz = next(iter(loaded.values()))["npz"]
    true_keys = [k for k in sample_npz.files if (k.startswith("td_") or k.startswith("modeldt_")) and k.endswith("_true")]
    if len(true_keys) == 0:
        raise RuntimeError("No '*_true' keys found in npz files; cannot determine time axis length.")
    sample_true = sample_npz[true_keys[0]]
    T = int(np.asarray(sample_true).ravel().shape[0])
    time = np.arange(T) * float(dt_for_timeaxis)

    # --------------------------
    # Figure: vertical stack to match your other figures
    # --------------------------
    n = len(test_dts)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    line_styles = ["-", "--", "-.", ":"]
    resume_table = []

    for ax, te_dt in zip(axes, test_dts):
        curves = []

        # Ground truth (try to find the exact matching true key; fallback to first true)
        gt_key = _find_key_for_test_dt_local(sample_npz, te_dt, "true")
        if gt_key is None:
            alt = [k for k in sample_npz.files if k.endswith("_true")]
            if len(alt) == 0:
                raise RuntimeError(f"No ground truth found for test dt {te_dt}")
            gt = np.asarray(sample_npz[alt[0]]).ravel()
            print(f"[plot_dt_timeseries] Warning: exact ground truth for dt={te_dt} not found; using {alt[0]}")
        else:
            gt = np.asarray(sample_npz[gt_key]).ravel()

        ax.plot(time[:gt.size], gt, color=COLORS["GT"], linestyle="--", lw=2.2, label=r"Ground Truth")
        curves.append(gt)

        # overlay predictions from each trained checkpoint
        for i, tr_dt in enumerate(train_dts):
            npz = loaded[tr_dt]["npz"]
            pred_key = _find_key_for_test_dt_local(npz, te_dt, "pred")
            if pred_key is None:
                print(f"[plot_dt_timeseries] Warning: no pred for test dt={te_dt} in train dt={tr_dt} (skipping)")
                continue
            pred = np.asarray(npz[pred_key]).ravel()
            L = min(pred.size, time.size)

            ax.plot(
                time[:L], pred[:L],
                color=COLORS["PhyGRU"],
                linestyle=line_styles[i % len(line_styles)],
                lw=1.8,
                alpha=0.95,
                label=rf"PhyGRU (train $\Delta t$={tr_dt:g})"
            )
            curves.append(pred[:L])

            mse_key = _find_key_for_test_dt_local(npz, te_dt, "mse")
            spk_key = _find_key_for_test_dt_local(npz, te_dt, "spearman")
            mse_val = float(npz[mse_key]) if (mse_key is not None and mse_key in npz.files) else np.nan
            spear_val = float(npz[spk_key]) if (spk_key is not None and spk_key in npz.files) else np.nan
            resume_table.append({
                "train_dt": tr_dt,
                "test_dt": te_dt,
                "mse": mse_val,
                "spearman": spear_val,
                "ckpt": loaded[tr_dt].get("path", None)
            })

        # axes styling consistent with your style
        apply_smart_ylim(ax, curves)
        ax.set_title(rf"Test $\Delta t$ = {te_dt}")
        ax.set_ylabel(r"$x(t)$")
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3)

        if ax is axes[0]:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel(r"$t\,[\mathrm{s}]$")

    plt.tight_layout()
    _ensure_dir(fig_dir)
    out_path = os.path.join(fig_dir, f"{dataset_name}_dt_timeseries_comparison.pdf")
    if save_fig:
        plt.savefig(out_path, bbox_inches="tight")
        print(f"[plot_dt_timeseries] Saved figure to {out_path}")
    plt.show()

    # compact resume table printout (same style you used elsewhere)
    print("\nCompact resume table (PhyGRU cross-eval):")
    print(" train_dt | test_dt |    mse    | spearman | ckpt")
    for r in resume_table:
        print(f" {r['train_dt']:7g} | {r['test_dt']:6g} | {r['mse']:8.3e} | {r['spearman']:8.3f} | {r['ckpt']}")

    return {
        "loaded": loaded,
        "time": time,
        "resume_table": resume_table
    }




# ============================================================
# EXECUTION (COMMENT / UNCOMMENT AS NEEDED)
# ============================================================

# --- TI ---
plot_TI_TV_datasets_overview(
    T=6000,
    dt=0.01,
    mode="TI",
    save_fig=True,
    fig_dir="figures"
)
plot_predictions("TI", "test", "TI")
plot_predictions("TI", "val",  "TI")

# --- TV ---
plot_TI_TV_datasets_overview(
    T=6000,
    dt=0.01,
    mode="TV",
    save_fig=True,
    fig_dir="figures"
)
plot_predictions("TV", "test", "TV")
plot_predictions("TV", "val",  "TV")

# --- TV_stack ---
plot_predictions("TV_stack", "test", "TV_stack")
plot_predictions("TV_stack", "val",  "TV_stack")

# --- UPDATE GATE ---
run_update_gate_analysis_TI(
    dataset_name="Sys_1",
    latent_dim=1,
    T=6000,
    dt=0.01,
    base_dir="TI",
    save_fig=True
)
run_update_gate_analysis_TI(
    dataset_name="Sys_3",
    latent_dim=2,
    T=6000,
    dt=0.01,
    base_dir="TI",
    save_fig=True
)

plot_identified_phygru_models(
    dataset_name="Sys_1",
    results_dir="TI/results_npz",
    dt=0.01,
    save_fig=True,
    fig_dir="figures"
)

compare_incremental_stability_phygru_vs_gru(
    dataset_name="Sys_1",
    latent_dim=1,
    T=6000,
    dt=0.01,
    N_PERTURB=100,
    perturb_range=3.0,
    base_dir="TI",
    fig_dir="./figures",
    save_fig=True,
    device="cpu",
    gru_hidden=32,
    colors=None
)


plot_dt_timeseries(results_dir=".\TI_dt\results_npz", fig_dir="figures", dataset_name="Sys_1",
                    train_dts=(0.01, 0.9), test_dts=(0.005, 0.01, 0.9), dt_for_timeaxis=0.01, save_fig=True)

#################################################

# !apt-get update -qq
# !apt-get install -y --no-install-recommends \
#     texlive-latex-base \
#     texlive-latex-extra \
#     texlive-fonts-recommended \
#     dvipng \
#     cm-super




# from google.colab import files
# import shutil

# folder_to_zip = "figures"
# zip_name = "figures.zip"

# shutil.make_archive("figures", 'zip', folder_to_zip)

# files.download(zip_name)


