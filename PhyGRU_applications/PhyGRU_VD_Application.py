import os
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

sns.set_style("whitegrid")
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})
sns.set_palette("colorblind")

print("Environment versions:")
print(f"Python:  {sys.version.split()[0]}")
print(f"torch:   {torch.__version__}")
print(f"numpy:   {np.__version__}")
print(f"seaborn: {sns.__version__}")

print("\nSystem hardware:")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Number of CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"   - Device {i}: {torch.cuda.get_device_name(i)}")
else:
    import platform
    print(f"  CPU: {platform.processor()}")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dt = 0.01
T  = 1000

INPUT_DIM  = 3
OUTPUT_DIM = 3

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_learnables(model, model_name="model"):
    print(f"\n{model_name} learnable parameters:")
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            print(f"  {name:40s} {tuple(param.shape)!s:18s} {n}")
    print(f"  {'TOTAL':40s} {'':18s} {total}")

def inference_time(model, u):
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        _ = model(u)
        t1 = time.time()
    B, TT, _ = u.shape
    return (t1 - t0) / (B * TT)

def standardize(train, val, test, eps=1e-8):
    mean = train.mean(dim=(0, 1), keepdim=True)
    std = train.std(dim=(0, 1), keepdim=True).clamp_min(eps)
    return (train - mean) / std, (val - mean) / std, (test - mean) / std, mean, std

def unstandardize(x, mean, std):
    return x * std + mean

def zero_last_linear(module):
    if isinstance(module, nn.Sequential):
        layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
        if len(layers) > 0:
            last = layers[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

class Vehicle7DOFSimulator:
    def __init__(self):
        self.m = 1500.0
        self.Iz = 2500.0
        self.Iw = 1.8
        self.lf = 1.25
        self.lr = 1.55
        self.tf = 1.60
        self.tr = 1.60
        self.Rw = 0.31
        self.Cf = 70000.0
        self.Cr = 80000.0
        self.Cd = 0.35
        self.Crr = 40.0
        self.Fx_max = 4500.0
        self.Fb_max = 9000.0
        self.T_drive_max = 3500.0
        self.T_brake_max = 6500.0
        self.v_min = 1.0

    def step(self, x, u):
        vx, vy, r, w_fl, w_fr, w_rl, w_rr = x
        delta, throttle, brake = u

        vx_safe = np.sign(vx) * max(abs(vx), self.v_min)
        if abs(vx_safe) < self.v_min:
            vx_safe = self.v_min

        alpha_f = delta - math.atan2(vy + self.lf * r, vx_safe)
        alpha_r = -math.atan2(vy - self.lr * r, vx_safe)

        Fyf = 2.0 * self.Cf * math.tanh(alpha_f)
        Fyr = 2.0 * self.Cr * math.tanh(alpha_r)

        Fx_drive = self.Fx_max * throttle
        Fx_brake = self.Fb_max * brake
        Fx_aero = self.Cd * vx * abs(vx)
        Fx_roll = self.Crr * math.tanh(vx)

        Fx_total = Fx_drive - Fx_brake - Fx_aero - Fx_roll

        vx_dot = (Fx_total - Fyf * math.sin(delta) + self.m * vy * r) / self.m
        vy_dot = (Fyf * math.cos(delta) + Fyr - self.m * vx * r) / self.m
        r_dot = (self.lf * Fyf * math.cos(delta) - self.lr * Fyr) / self.Iz

        ax = vx_dot - vy * r
        ay = vy_dot + vx * r

        T_drive = self.T_drive_max * throttle
        T_brake = self.T_brake_max * brake

        Fx_rear = 0.6 * Fx_total
        Fx_front = 0.4 * Fx_total

        w_fl_dot = (-T_brake - self.Rw * (Fx_front / 2.0)) / self.Iw
        w_fr_dot = (-T_brake - self.Rw * (Fx_front / 2.0)) / self.Iw
        w_rl_dot = ((T_drive - T_brake) - self.Rw * (Fx_rear / 2.0)) / self.Iw
        w_rr_dot = ((T_drive - T_brake) - self.Rw * (Fx_rear / 2.0)) / self.Iw

        x_next = np.array([
            vx + dt * vx_dot,
            vy + dt * vy_dot,
            r + dt * r_dot,
            w_fl + dt * w_fl_dot,
            w_fr + dt * w_fr_dot,
            w_rl + dt * w_rl_dot,
            w_rr + dt * w_rr_dot,
        ], dtype=np.float32)

        y = np.array([ax, ay, r], dtype=np.float32)
        return x_next, y

def smooth_random_controls(T, rng):
    t = np.arange(T) * dt

    f1 = rng.uniform(0.05, 0.20)
    f2 = rng.uniform(0.15, 0.45)
    p1 = rng.uniform(0, 2 * np.pi)
    p2 = rng.uniform(0, 2 * np.pi)
    delta = 0.06 * np.sin(2 * np.pi * f1 * t + p1) + 0.03 * np.sin(2 * np.pi * f2 * t + p2)

    ft = rng.uniform(0.03, 0.15)
    pt = rng.uniform(0, 2 * np.pi)
    throttle = 0.55 + 0.20 * np.sin(2 * np.pi * ft * t + pt)
    throttle += 0.05 * np.sin(2 * np.pi * (ft * 2.0) * t + 0.3)
    throttle = np.clip(throttle, 0.0, 1.0)

    fb = rng.uniform(0.02, 0.10)
    pb = rng.uniform(0, 2 * np.pi)
    brake = 0.10 * np.maximum(0.0, np.sin(2 * np.pi * fb * t + pb))
    brake += 0.03 * np.maximum(0.0, np.sin(2 * np.pi * (fb * 2.5) * t + 1.0))
    brake = np.clip(brake, 0.0, 1.0)

    return np.stack([delta, throttle, brake], axis=1).astype(np.float32)

def simulate_sequence(u_seq, noise_std=(0.15, 0.15, 0.01), seed=0):
    rng = np.random.default_rng(seed)
    sim = Vehicle7DOFSimulator()

    x = np.array([15.0, 0.0, 0.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32)
    pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    Y_noisy = []
    Y_clean = []
    P = []

    for t in range(u_seq.shape[0]):
        vx, vy, r = x[0], x[1], x[2]

        x, y_clean = sim.step(x, u_seq[t])

        noise = rng.normal(0.0, np.array(noise_std, dtype=np.float32), size=(3,)).astype(np.float32)
        y_noisy = y_clean + noise

        Y_clean.append(y_clean)
        Y_noisy.append(y_noisy)

        psi = pose[2]
        x_dot = vx * np.cos(psi) - vy * np.sin(psi)
        y_dot = vx * np.sin(psi) + vy * np.cos(psi)
        psi_dot = r

        pose = pose + dt * np.array([x_dot, y_dot, psi_dot], dtype=np.float32)
        P.append(pose.copy())

    return (
        np.stack(Y_noisy, axis=0).astype(np.float32),
        np.stack(Y_clean, axis=0).astype(np.float32),
        np.stack(P, axis=0).astype(np.float32),
    )

def make_dataset(n_seq=128, T=1000, seed=0):
    rng = np.random.default_rng(seed)
    U = np.zeros((n_seq, T, INPUT_DIM), dtype=np.float32)
    Y_noisy = np.zeros((n_seq, T, OUTPUT_DIM), dtype=np.float32)
    Y_clean = np.zeros((n_seq, T, OUTPUT_DIM), dtype=np.float32)
    P = np.zeros((n_seq, T, 3), dtype=np.float32)

    for i in range(n_seq):
        u_seq = smooth_random_controls(T, rng)
        y_noisy_seq, y_clean_seq, pose_seq = simulate_sequence(u_seq, seed=seed + i)
        U[i] = u_seq
        Y_noisy[i] = y_noisy_seq
        Y_clean[i] = y_clean_seq
        P[i] = pose_seq

    return torch.tensor(U), torch.tensor(Y_noisy), torch.tensor(Y_clean), torch.tensor(P)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, u):
        h, _ = self.gru(u)
        return self.fc(h)

class BicyclePrior(nn.Module):
    def __init__(self, y_mean, y_std):
        super().__init__()
        self.m = 1500.0
        self.Iz = 2500.0
        self.lf = 1.25
        self.lr = 1.55
        self.Cf = 65000.0
        self.Cr = 70000.0
        self.Cd = 0.30
        self.Crr = 30.0
        self.Fx_max = 4200.0
        self.Fb_max = 8500.0
        self.v_min = 1.0

        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32).view(1, 3))
        self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32).view(1, 3))

    def forward(self, state, u):
        vx = state[:, 0]
        vy = state[:, 1]
        r = state[:, 2]
        delta = u[:, 0]
        throttle = u[:, 1]
        brake = u[:, 2]

        vx_safe = torch.where(vx.abs() < self.v_min, torch.ones_like(vx) * self.v_min, vx)

        alpha_f = delta - torch.atan2(vy + self.lf * r, vx_safe)
        alpha_r = -torch.atan2(vy - self.lr * r, vx_safe)

        Fyf = 2.0 * self.Cf * torch.tanh(alpha_f)
        Fyr = 2.0 * self.Cr * torch.tanh(alpha_r)

        Fx = self.Fx_max * throttle - self.Fb_max * brake - self.Cd * vx * vx.abs() - self.Crr * torch.tanh(vx)

        vx_dot = (Fx - Fyf * torch.sin(delta) + self.m * vy * r) / self.m
        vy_dot = (Fyf * torch.cos(delta) + Fyr - self.m * vx * r) / self.m
        r_dot = (self.lf * Fyf * torch.cos(delta) - self.lr * Fyr) / self.Iz

        ax = vx_dot - vy * r
        ay = vy_dot + vx * r

        y_phys = torch.stack([ax, ay, r], dim=1)
        y_phys = (y_phys - self.y_mean) / self.y_std

        s_dot = torch.stack([vx_dot, vy_dot, r_dot], dim=1)
        return s_dot, y_phys

class PhyGRUResidualCell(nn.Module):
    def __init__(self, physics_prior, input_dim=3, latent_dim=4):
        super().__init__()
        self.prior = physics_prior
        self.input_dim = input_dim
        self.state_dim = 3
        self.latent_dim = latent_dim

        total_state = self.state_dim + self.latent_dim

        self.z_gate = nn.Sequential(
            nn.Linear(total_state + input_dim, total_state),
            nn.Sigmoid()
        )

        if latent_dim > 0:
            self.latent_dyn = nn.Sequential(
                nn.Linear(total_state + input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, latent_dim)
            )
        else:
            self.latent_dyn = None

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.state_dim + input_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
            for _ in range(3)
        ])

        for branch in self.branches:
            zero_last_linear(branch)

        with torch.no_grad():
            last_linear = None
            for m in self.z_gate.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None and last_linear.bias is not None:
                last_linear.bias.fill_(2.0)

    def forward(self, state, u):
        s = state[:, :self.state_dim]
        latent = state[:, self.state_dim:]

        s_dot_phys, y_phys = self.prior(s, u)
        s_phys_next = s + dt * s_dot_phys

        if self.latent_dim > 0:
            latent_dot = self.latent_dyn(torch.cat([state, u], dim=1))
            latent_next = latent + dt * latent_dot
            candidate = torch.cat([s_phys_next, latent_next], dim=1)
        else:
            candidate = s_phys_next

        z = self.z_gate(torch.cat([state, u], dim=1))
        next_state = z * candidate + (1.0 - z) * state

        branch_in = torch.cat([s, u], dim=1)
        residuals = torch.cat([head(branch_in) for head in self.branches], dim=1)
        y = y_phys + residuals

        return next_state, y

class PhyGRUResidual(nn.Module):
    def __init__(self, y_mean, y_std, input_dim=3, latent_dim=4, init_state=(15.0, 0.0, 0.0)):
        super().__init__()
        prior = BicyclePrior(y_mean=y_mean, y_std=y_std)
        self.cell = PhyGRUResidualCell(prior, input_dim=input_dim, latent_dim=latent_dim)
        self.state_dim = 3
        self.latent_dim = latent_dim
        self.register_buffer("init_state", torch.tensor(init_state, dtype=torch.float32))

    def forward(self, u_seq):
        B, Tt, _ = u_seq.shape
        state = self.init_state.unsqueeze(0).repeat(B, 1).to(u_seq.device, u_seq.dtype)
        if self.latent_dim > 0:
            latent0 = torch.zeros(B, self.latent_dim, device=u_seq.device, dtype=u_seq.dtype)
            state = torch.cat([state, latent0], dim=1)

        ys = []
        for t in range(Tt):
            state, y = self.cell(state, u_seq[:, t])
            ys.append(y)
        return torch.stack(ys, dim=1)


class BicyclePhysicsLaw(nn.Module):
    def __init__(self, y_mean, y_std):
        super().__init__()
        self.prior = BicyclePrior(y_mean, y_std)

    def forward(self, state, u):
        # state = [vx, vy, r]
        s_dot, _ = self.prior(state, u)
        return s_dot

class PhyGRUBaseCell(nn.Module):
    def __init__(self, state_dim, input_dim, physics_law, latent_dim=0):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.physics_law = physics_law

        total_state = state_dim + latent_dim

        if latent_dim > 0:
            self.latent_dyn = nn.Linear(total_state + input_dim, latent_dim)
        else:
            self.latent_dyn = None

        self.z_gate = nn.Sequential(
            nn.Linear(total_state + input_dim, total_state),
            nn.Sigmoid()
        )

    def forward(self, state, u):
        # --- physics ---
        phys_dot = self.physics_law(state[:, :self.state_dim], u)
        phys_next = state[:, :self.state_dim] + dt * phys_dot

        # --- latent ---
        if self.latent_dim > 0:
            latent = state[:, self.state_dim:]
            latent_dot = self.latent_dyn(torch.cat([state, u], dim=1))
            latent_next = latent + dt * latent_dot
            candidate = torch.cat([phys_next, latent_next], dim=1)
        else:
            candidate = phys_next

        # --- GRU update ---
        z = self.z_gate(torch.cat([state, u], dim=1))
        next_state = z * candidate + (1 - z) * state

        return next_state


class PhyGRUBase(nn.Module):
    def __init__(self, y_mean, y_std, state_dim=3, input_dim=3, latent_dim=4,
                 init_state=(15.0, 0.0, 0.0)):
        super().__init__()

        self.physics_law = BicyclePhysicsLaw(y_mean, y_std)

        self.cell = PhyGRUBaseCell(
            state_dim=state_dim,
            input_dim=input_dim,
            physics_law=self.physics_law,
            latent_dim=latent_dim
        )

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.register_buffer(
            "init_state",
            torch.tensor(init_state, dtype=torch.float32)
        )

        self.output_prior = BicyclePrior(y_mean, y_std)

    def forward(self, u_seq):
        B, Tt, _ = u_seq.shape

        state = self.init_state.unsqueeze(0).repeat(B, 1).to(u_seq.device, u_seq.dtype)

        if self.latent_dim > 0:
            latent0 = torch.zeros(B, self.latent_dim, device=u_seq.device, dtype=u_seq.dtype)
            state = torch.cat([state, latent0], dim=1)

        ys = []
        for t in range(Tt):
            state = self.cell(state, u_seq[:, t])


            _, y_phys = self.output_prior(state[:, :self.state_dim], u_seq[:, t])
            ys.append(y_phys)

        return torch.stack(ys, dim=1)

def train_model(model, U_train, Y_train, U_val, Y_val, epochs=100, lr=1e-3, batch_size=8, verbose=True):
    model = model.to(device)
    U_train = U_train.to(device)
    Y_train = Y_train.to(device)
    U_val = U_val.to(device)
    Y_val = Y_val.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_train = U_train.shape[0]
    idx = np.arange(n_train)
    best_val = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(idx)
        train_losses = []

        for start in range(0, n_train, batch_size):
            batch_idx = idx[start:start + batch_size]
            u_b = U_train[batch_idx]
            y_b = Y_train[batch_idx]

            opt.zero_grad()
            pred = model(u_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(U_val)
            val_loss = loss_fn(val_pred, Y_val).item()

        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch <= 5 or epoch % 10 == 0):
            print(f"Epoch {epoch:03d} | train={train_loss:.4e} | val={val_loss:.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val

def evaluate(model, U, Y):
    model.eval()
    with torch.no_grad():
        pred = model(U.to(device)).cpu()
    mse = ((pred - Y) ** 2).mean().item()
    per_channel = ((pred - Y) ** 2).mean(dim=(0, 1)).numpy()
    return pred, mse, per_channel

def plot_sample(y_true, y_pred, title=""):
    labels = ["ax", "ay", "yaw_rate"]
    t = np.arange(y_true.shape[0]) * dt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, y_true[:, i], label="true", lw=2.0)
        ax.plot(t, y_pred[:, i], label="pred", lw=2.0)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.35)
        if i == 0:
            ax.set_title(title)
        if i == 2:
            ax.set_xlabel("time [s]")
    sns.despine(fig=fig)
    plt.show()

def plot_gt_all_models(y_true, y_gru, y_phy_base, y_phy_resid, title="Ground Truth vs GRU vs PhyGRU Base vs PhyGRU Residual"):
    labels = ["ax", "ay", "yaw_rate"]
    t = np.arange(y_true.shape[0]) * dt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, y_true[:, i], label="Ground Truth", lw=2.2)
        ax.plot(t, y_gru[:, i], label="GRU", lw=2.0)
        ax.plot(t, y_phy_base[:, i], label="PhyGRU Base", lw=2.0)
        ax.plot(t, y_phy_resid[:, i], label="PhyGRU Residual", lw=2.0)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.35)
        if i == 0:
            ax.set_title(title)
        if i == 2:
            ax.set_xlabel("time [s]")
    axes[0].legend(loc="best")
    sns.despine(fig=fig)
    plt.show()

def reconstruct_pose_from_outputs(y_seq, init_vx=15.0, init_vy=0.0, init_pose=(0.0, 0.0, 0.0)):
    x, y, psi = init_pose
    vx, vy = init_vx, init_vy

    traj = []
    for k in range(y_seq.shape[0]):
        ax_b, ay_b, r = y_seq[k]

        vx = vx + dt * ax_b
        vy = vy + dt * ay_b
        psi = psi + dt * r

        x = x + dt * (vx * np.cos(psi) - vy * np.sin(psi))
        y = y + dt * (vx * np.sin(psi) + vy * np.cos(psi))

        traj.append([x, y, psi])

    return np.asarray(traj, dtype=np.float32)

def _add_gradient_path(ax, x, y, base_color, label=None, lw=2.5, alpha_min=0.15, alpha_max=1.0):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    rgba = np.array(to_rgba(base_color))
    colors = np.tile(rgba, (len(segments), 1))
    colors[:, 3] = np.linspace(alpha_min, alpha_max, len(segments))

    lc = LineCollection(segments, colors=colors, linewidths=lw, zorder=2)
    ax.add_collection(lc)

    ax.scatter([x[0]], [y[0]], s=50, color=base_color, marker="o", zorder=3)
    ax.scatter([x[-1]], [y[-1]], s=50, color=base_color, marker="s", zorder=3)

    return Line2D([0], [0], color=base_color, lw=lw, label=label)

def plot_xy_and_controls_all(y_gt, y_gru, y_phy_base, y_phy_resid, controls, title="XY map and control signals"):
    pose_gt = reconstruct_pose_from_outputs(y_gt)
    pose_gru = reconstruct_pose_from_outputs(y_gru)
    pose_base = reconstruct_pose_from_outputs(y_phy_base)
    pose_resid = reconstruct_pose_from_outputs(y_phy_resid)

    t = np.arange(controls.shape[0]) * dt

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.0, 1.0, 1.0], hspace=0.28)

    ax_xy = fig.add_subplot(gs[0])
    ax_steer = fig.add_subplot(gs[1], sharex=None)
    ax_throttle = fig.add_subplot(gs[2], sharex=ax_steer)
    ax_brake = fig.add_subplot(gs[3], sharex=ax_steer)

    palette = sns.color_palette("colorblind", 4)
    c_gt, c_gru, c_base, c_resid = palette[0], palette[1], palette[2], palette[3]

    _add_gradient_path(ax_xy, pose_gt[:, 0], pose_gt[:, 1], c_gt, label="Ground Truth")
    _add_gradient_path(ax_xy, pose_gru[:, 0], pose_gru[:, 1], c_gru, label="GRU")
    _add_gradient_path(ax_xy, pose_base[:, 0], pose_base[:, 1], c_base, label="PhyGRU Base")
    _add_gradient_path(ax_xy, pose_resid[:, 0], pose_resid[:, 1], c_resid, label="PhyGRU Residual")

    ax_xy.set_title(title)
    ax_xy.grid(True, alpha=0.35)
    ax_xy.axis("equal")
    ax_xy.legend(loc="best")

    ax_steer.plot(t, controls[:, 0], color=palette[0], lw=1.8)
    ax_steer.set_ylabel("steering")
    ax_steer.grid(True, alpha=0.35)

    ax_throttle.plot(t, controls[:, 1], color=palette[1], lw=1.8)
    ax_throttle.set_ylabel("throttle")
    ax_throttle.grid(True, alpha=0.35)

    ax_brake.plot(t, controls[:, 2], color=palette[2], lw=1.8)
    ax_brake.set_ylabel("brake")
    ax_brake.set_xlabel("time [s]")
    ax_brake.grid(True, alpha=0.35)

    sns.despine(fig=fig)
    plt.show()

def plot_trajectories_grid_all(y_gt_all, y_gru_all, y_phy_base_all, y_phy_resid_all, nrows=6, ncols=6):
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 14))

    palette = sns.color_palette("colorblind", 4)
    c_gt, c_gru, c_base, c_resid = palette[0], palette[1], palette[2], palette[3]

    for i, ax in enumerate(axes.flat):
        if i >= y_gt_all.shape[0]:
            ax.axis("off")
            continue

        pose_gt = reconstruct_pose_from_outputs(y_gt_all[i])
        pose_gru = reconstruct_pose_from_outputs(y_gru_all[i])
        pose_base = reconstruct_pose_from_outputs(y_phy_base_all[i])
        pose_resid = reconstruct_pose_from_outputs(y_phy_resid_all[i])

        ax.plot(pose_gt[:, 0],     pose_gt[:, 1],     color=c_gt,     lw=1.2)
        ax.plot(pose_gru[:, 0],    pose_gru[:, 1],    color=c_gru,    lw=1.0)
        ax.plot(pose_base[:, 0],   pose_base[:, 1],   color=c_base,   lw=1.0)
        ax.plot(pose_resid[:, 0],  pose_resid[:, 1],  color=c_resid,  lw=1.0)

        x_all = np.concatenate([pose_gt[:, 0], pose_gru[:, 0], pose_base[:, 0], pose_resid[:, 0]])
        y_all = np.concatenate([pose_gt[:, 1], pose_gru[:, 1], pose_base[:, 1], pose_resid[:, 1]])

        x_mid = 0.5 * (x_all.min() + x_all.max())
        y_mid = 0.5 * (y_all.min() + y_all.max())
        max_range = max(x_all.max() - x_all.min(), y_all.max() - y_all.min()) / 2

        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

    plt.show()

if __name__ == "__main__":
    sns.set_palette("colorblind")

    N_train = 250
    N_val   = 50
    N_test  = 36

    print("\nGenerating datasets from 7DOF simulator...")
    U_train, Y_train, _, _ = make_dataset(N_train, T=T, seed=1)
    U_val, Y_val, _, _ = make_dataset(N_val, T=T, seed=1001)
    U_test, Y_test, Y_test_clean, P_test = make_dataset(N_test, T=T * 5, seed=2001)

    Y_train_n, Y_val_n, Y_test_n, Y_mean, Y_std = standardize(Y_train, Y_val, Y_test)

    y_mean_np = Y_mean.squeeze(0).squeeze(0).numpy()
    y_std_np  = Y_std.squeeze(0).squeeze(0).numpy()

    print("\nTraining baseline GRU...")
    gru = GRUModel(INPUT_DIM, hidden_dim=64, output_dim=OUTPUT_DIM)
    print_learnables(gru, "GRU")
    gru, gru_hist, gru_best = train_model(
        gru, U_train, Y_train_n, U_val, Y_val_n,
        epochs=120, lr=1e-3, batch_size=8, verbose=True
    )

    print("\nTraining PhyGRU base (bicycle prior, no residual branches)...")
    phygru_base = PhyGRUBase(
        y_mean=y_mean_np,
        y_std=y_std_np,
        input_dim=INPUT_DIM,
        latent_dim=4,
        init_state=(15.0, 0.0, 0.0)
    )
    print_learnables(phygru_base, "PhyGRU Base")
    phygru_base, base_hist, base_best = train_model(
        phygru_base, U_train, Y_train_n, U_val, Y_val_n,
        epochs=10, lr=1e-3, batch_size=8, verbose=True
    )

    print("\nTraining PhyGRU residual (bicycle prior + residual branches)...")
    phygru_resid = PhyGRUResidual(
        y_mean=y_mean_np,
        y_std=y_std_np,
        input_dim=INPUT_DIM,
        latent_dim=4,
        init_state=(15.0, 0.0, 0.0)
    )
    print_learnables(phygru_resid, "PhyGRU Residual")
    phygru_resid, resid_hist, resid_best = train_model(
        phygru_resid, U_train, Y_train_n, U_val, Y_val_n,
        epochs=10, lr=1e-3, batch_size=8, verbose=True
    )

    gru_pred_n, gru_mse_n, gru_perch_n = evaluate(gru, U_test, Y_test_n)
    base_pred_n, base_mse_n, base_perch_n = evaluate(phygru_base, U_test, Y_test_n)
    resid_pred_n, resid_mse_n, resid_perch_n = evaluate(phygru_resid, U_test, Y_test_n)

    gru_pred   = unstandardize(gru_pred_n, Y_mean, Y_std)
    base_pred  = unstandardize(base_pred_n, Y_mean, Y_std)
    resid_pred = unstandardize(resid_pred_n, Y_mean, Y_std)

    gru_mse = ((gru_pred - Y_test) ** 2).mean().item()
    base_mse = ((base_pred - Y_test) ** 2).mean().item()
    resid_mse = ((resid_pred - Y_test) ** 2).mean().item()

    gru_perch = ((gru_pred - Y_test) ** 2).mean(dim=(0, 1)).numpy()
    base_perch = ((base_pred - Y_test) ** 2).mean(dim=(0, 1)).numpy()
    resid_perch = ((resid_pred - Y_test) ** 2).mean(dim=(0, 1)).numpy()

    print("\n==============================")
    print("TEST RESULTS")
    print("==============================")
    print(f"GRU            | MSE = {gru_mse:.4e} | per-channel = {gru_perch}")
    print(f"PhyGRU Base    | MSE = {base_mse:.4e} | per-channel = {base_perch}")
    print(f"PhyGRU Residual| MSE = {resid_mse:.4e} | per-channel = {resid_perch}")

    gru_inf = inference_time(gru, U_test.to(device)) * 1000
    base_inf = inference_time(phygru_base, U_test.to(device)) * 1000
    resid_inf = inference_time(phygru_resid, U_test.to(device)) * 1000

    print(f"\nInference per sample-step:")
    print(f"GRU            : {gru_inf:.4f} ms")
    print(f"PhyGRU Base    : {base_inf:.4f} ms")
    print(f"PhyGRU Residual: {resid_inf:.4f} ms")

    sample_id = 0

    plot_gt_all_models(
        Y_test[sample_id].numpy(),
        gru_pred[sample_id].numpy(),
        base_pred[sample_id].numpy(),
        resid_pred[sample_id].numpy(),
        title=" "
    )

    y_gt     = Y_test_clean[sample_id].numpy()
    y_gru    = gru_pred[sample_id].numpy()
    y_base   = base_pred[sample_id].numpy()
    y_resid  = resid_pred[sample_id].numpy()
    controls = U_test[sample_id].numpy()

    plot_xy_and_controls_all(
        y_gt=y_gt,
        y_gru=y_gru,
        y_phy_base=y_base,
        y_phy_resid=y_resid,
        controls=controls,
        title=" "
    )

    plot_trajectories_grid_all(
        y_gt_all=Y_test_clean.numpy(),
        y_gru_all=gru_pred.numpy(),
        y_phy_base_all=base_pred.numpy(),
        y_phy_resid_all=resid_pred.numpy()
    )







