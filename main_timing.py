# ============================================================
# INFERENCE-TIME BENCHMARK
# Structured table output
# GRU (manual, torch-matched), PhyGRU, PhyGRU_rg
# ============================================================

import torch
import torch.nn as nn
import time
import math
import random
import numpy as np
import sys
import platform

# ============================================================
# Environment info
# ============================================================
print("Environment versions:")
print(f"Python:  {sys.version.split()[0]}")
print(f"torch:   {torch.__version__}")
print(f"numpy:   {np.__version__}")

print("\nSystem hardware:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"CPU: {platform.processor()}")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Benchmark settings
# ============================================================
dt = 0.01
T  = 6000
N_RUNS = 15

# ============================================================
# Utilities
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def inference_time_stats(model, u, n_runs=N_RUNS):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(u)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            B, TT, _ = u.shape
            times.append((t1 - t0) / (B * TT))
    times = np.array(times)
    return times.mean(), times.std()

# ============================================================
# Dataset (Sys_1)
# ============================================================
def generate_data(u_fn):
    x, xd = 0.0, 0.0
    xs, us = [], []
    for t in range(T):
        u = u_fn(t)
        xdd = (u - 0.5 * xd - 0.2 * x)
        xd += dt * xdd
        x  += dt * xd
        xs.append([x])
        us.append([u])
    return torch.tensor(xs), torch.tensor(us)

x_test, u_test = generate_data(
    lambda t: math.tanh((0.5-0.00005*t)*((0.30-0.002*t)*math.sin((0.00050+0.0000005*t)*t)))
)

x_test = x_test.unsqueeze(0).to(device)
u_test = u_test.unsqueeze(0).to(device)

x_test /= torch.max(torch.abs(x_test))
u_test /= torch.max(torch.abs(u_test))

# ============================================================
# Manual GRU (Torch-parameter matched)
# ============================================================
class GRUCellTorchLike(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_iz = nn.Linear(input_dim, hidden_dim)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim)
        self.W_ir = nn.Linear(input_dim, hidden_dim)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim)
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, x):
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h))
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h))
        return (1 - z) * h + z * n

class GRUManual(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = GRUCellTorchLike(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, u_seq):
        B, Tt, _ = u_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=u_seq.device)
        ys = []
        for t in range(Tt):
            h = self.cell(h, u_seq[:, t])
            ys.append(self.fc(h))
        return torch.stack(ys, dim=1)

# ============================================================
# Physics model
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
# PhyGRU (no reset gate)
# ============================================================
class PhyGRUCell(nn.Module):
    def __init__(self, state_dim, input_dim, physics_law, latent_dim=0):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        total = state_dim + latent_dim
        self.physics = physics_law
        if latent_dim > 0:
            self.latent_dyn = nn.Linear(total + input_dim, latent_dim)
        else:
            self.latent_dyn = None
        self.z_gate = nn.Sequential(
            nn.Linear(total + input_dim, total),
            nn.Sigmoid()
        )

    def forward(self, state, u):
        phys_dot = self.physics(state[:, :self.state_dim], u)
        phys_next = state[:, :self.state_dim] + dt * phys_dot
        if self.latent_dim > 0:
            latent = state[:, self.state_dim:]
            latent_dot = self.latent_dyn(torch.cat([state, u], dim=1))
            latent_next = latent + dt * latent_dot
            candidate = torch.cat([phys_next, latent_next], dim=1)
        else:
            candidate = phys_next
        z = self.z_gate(torch.cat([state, u], dim=1))
        return z * candidate + (1 - z) * state

class PhyGRU(nn.Module):
    def __init__(self, physics_law, state_dim, input_dim, latent_dim=0):
        super().__init__()
        self.cell = PhyGRUCell(state_dim, input_dim, physics_law, latent_dim)
        self.state_dim = state_dim
        self.latent_dim = latent_dim

    def forward(self, u_seq):
        B, Tt, _ = u_seq.shape
        state = torch.zeros(B, self.state_dim + self.latent_dim, device=u_seq.device)
        ys = []
        for t in range(Tt):
            state = self.cell(state, u_seq[:, t])
            ys.append(state[:, :1])
        return torch.stack(ys, dim=1)

# ============================================================
# PhyGRU with reset gate (PhyGRU_rg)
# ============================================================
class PhyGRUCellRG(nn.Module):
    def __init__(self, state_dim, input_dim, physics_law, latent_dim=0):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.physics_law = physics_law
        total_state = state_dim + latent_dim
        if latent_dim > 0:
            self.latent_dyn = nn.Linear(total_state + input_dim, latent_dim)
            self.r_gate = nn.Sequential(
                nn.Linear(total_state + input_dim, latent_dim),
                nn.Sigmoid()
            )
        else:
            self.latent_dyn = None
            self.r_gate = None
        self.z_gate = nn.Sequential(
            nn.Linear(total_state + input_dim, total_state),
            nn.Sigmoid()
        )

    def forward(self, state, u):
        phys_dot = self.physics_law(state[:, :self.state_dim], u)
        phys_next = state[:, :self.state_dim] + dt * phys_dot
        if self.latent_dim > 0:
            latent = state[:, self.state_dim:]
            r = self.r_gate(torch.cat([state, u], dim=1))
            gated_latent = r * latent
            gated_state = torch.cat([state[:, :self.state_dim], gated_latent], dim=1)
            latent_dot = self.latent_dyn(torch.cat([gated_state, u], dim=1))
            latent_next = latent + dt * latent_dot
            candidate = torch.cat([phys_next, latent_next], dim=1)
        else:
            candidate = phys_next
        z = self.z_gate(torch.cat([state, u], dim=1))
        return z * candidate + (1 - z) * state

class PhyGRU_rg(nn.Module):
    def __init__(self, physics_law, state_dim, input_dim, latent_dim=0):
        super().__init__()
        self.cell = PhyGRUCellRG(state_dim, input_dim, physics_law, latent_dim)
        self.state_dim = state_dim
        self.latent_dim = latent_dim

    def forward(self, u_seq):
        B, Tt, _ = u_seq.shape
        state = torch.zeros(B, self.state_dim + self.latent_dim, device=u_seq.device)
        ys = []
        for t in range(Tt):
            state = self.cell(state, u_seq[:, t])
            ys.append(state[:, :1])
        return torch.stack(ys, dim=1)

# ============================================================
# Benchmark table
# ============================================================
rows = []

hidden_sizes = [1, 2, 4, 8, 32]
latent_dims  = [0, 1, 2, 3]

for h in hidden_sizes:
    model = GRUManual(1, h).to(device)
    mean_t, std_t = inference_time_stats(model, u_test)
    rows.append(("GRU", h, count_parameters(model), mean_t*1000, std_t*1000))

for ld in latent_dims:
    model = PhyGRU(MassSpringDamperLaw(), 2, 1, latent_dim=ld).to(device)
    mean_t, std_t = inference_time_stats(model, u_test)
    rows.append(("PhyGRU", ld, count_parameters(model), mean_t*1000, std_t*1000))

for ld in latent_dims:
    model = PhyGRU_rg(MassSpringDamperLaw(), 2, 1, latent_dim=ld).to(device)
    mean_t, std_t = inference_time_stats(model, u_test)
    rows.append(("PhyGRU_rg", ld, count_parameters(model), mean_t*1000, std_t*1000))

# ============================================================
# Print table
# ============================================================
print("\n===== INFERENCE TIME SUMMARY =====")
print(f"{'Model':<10} {'Hidden/Latent':<15} {'Params':<8} {'Mean (ms)':<12} {'Std (ms)':<10}")
print("-"*60)
for r in rows:
    print(f"{r[0]:<10} {r[1]:<15d} {r[2]:<8d} {r[3]:<12.4f} {r[4]:<10.4f}")

print("\nDone.")

