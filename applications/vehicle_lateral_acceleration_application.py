import copy
import math
import random
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Seeds & Type Aliases
# ============================================================
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

Tensor = torch.Tensor
PhysicsFn = Callable[[Tensor, Tensor], Tensor]

# ============================================================
# Utilities
# ============================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _activation(name: str) -> nn.Module:
    name = name.lower()
    activations = {
        "tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(),
        "silu": nn.SiLU(), "swish": nn.SiLU(), "elu": nn.ELU(),
        "lrelu": nn.LeakyReLU(0.01), "leaky_relu": nn.LeakyReLU(0.01)
    }
    if name in activations:
        return activations[name]
    raise ValueError(f"Unknown activation '{name}'")

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, num_layers=2, activation="tanh", bias=True):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_layers == 1:
            self.net = nn.Linear(in_features, out_features, bias=bias)
            return
        hidden_features = hidden_features or max(in_features, out_features)
        layers = []
        dim_in = in_features
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim_in, hidden_features, bias=bias))
            layers.append(_activation(activation))
            dim_in = hidden_features
        layers.append(nn.Linear(dim_in, out_features, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

# ============================================================
# Tensor scaler
# ============================================================
class TensorScaler:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        self.std = x.std(dim=(0, 1), keepdim=True).clamp_min(self.eps)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean

# ============================================================
# Dataset Logic
# ============================================================
@dataclass
class VehicleTrueParams:
    mass: float = 1500.0
    iz: float = 2250.0
    lf: float = 1.2
    lr: float = 1.6
    cf: float = 70000
    cr: float = 80000
    vx_min: float = 4.0
    mu: float = 0.85
    g: float = 9.81
    aero_yaw: float = 300.0
    noise_std: float = 0.15

def simulate_sequence(T, dt, params, shift=0.0):
    vy = 0.0
    r = 0.0
    u, y = [], []

    Fmax_f = params.mu * params.mass * params.g * params.lr / (params.lf + params.lr)
    Fmax_r = params.mu * params.mass * params.g * params.lf / (params.lf + params.lr)

    for k in range(T):
        ramp = 0.0 if T <= 1 else (k / (T - 1))

        delta = ramp * (0.15 * np.sin(0.02 * k + shift) + 0.05 * np.sin(0.07 * k + shift * 1.5))
        vx = 15.0 + ramp * (5 * np.sin(0.01 * k + shift * 0.5))

        alpha_f = delta - (vy + params.lf * r) / vx
        alpha_r = -(vy - params.lr * r) / vx

        Fyf = Fmax_f * np.tanh((params.cf / Fmax_f) * alpha_f)
        Fyr = Fmax_r * np.tanh((params.cr / Fmax_r) * alpha_r)

        vy_dot = (Fyf + Fyr) / params.mass - vx * r
        r_dot = (params.lf * Fyf - params.lr * Fyr - params.aero_yaw * r * np.abs(r)) / params.iz

        vy += dt * vy_dot
        r += dt * r_dot

        noise = 0.0 if k == 0 else np.random.normal(0, params.noise_std)
        ay = vy_dot + vx * r + noise

        u.append([delta, vx])
        y.append([ay])

    return np.array(u, dtype=np.float32), np.array(y, dtype=np.float32)

def generate_dataset(N, T, dt, shift=0.0):
    params = VehicleTrueParams()
    U, Y = [], []
    for i in range(N):
        u, y = simulate_sequence(T, dt, params, shift + (i * 0.1))
        U.append(u)
        Y.append(y)
    return torch.tensor(np.array(U)), torch.tensor(np.array(Y))

# ============================================================
# PhyGRU cell (Standard Version)
# ============================================================
class PhyGRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        physics_prior: Optional[Union[PhysicsFn, nn.Module]],
        *,
        physical_size: Optional[int] = None,
        latent_size: int = 0,
        dt: float = 1.0,
        latent_type: Literal["linear", "mlp"] = "linear",
        latent_activation: str = "tanh",
        latent_hidden_size: Optional[int] = None,
        latent_num_layers: int = 2,
        gate_bias: bool = True,
        latent_mode: Literal["gate", "residual"] = "gate",
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        if state_size <= 0:
            raise ValueError("state_size must be > 0")

        self.input_size = input_size
        self.state_size = state_size
        self.physical_size = physical_size if physical_size is not None else state_size
        self.latent_size = latent_size
        self.total_size = self.physical_size + self.latent_size
        self.dt = float(dt)
        self.physics_prior = physics_prior
        self.latent_mode = latent_mode

        self.gate = nn.Linear(self.total_size + input_size, self.total_size, bias=gate_bias)

        if self.latent_size > 0:
            latent_in = self.total_size + input_size
            if latent_type == "linear":
                self.latent_dyn = nn.Linear(latent_in, self.latent_size)
            elif latent_type == "mlp":
                self.latent_dyn = MLP(
                    latent_in,
                    self.latent_size,
                    hidden_features=latent_hidden_size,
                    num_layers=latent_num_layers,
                    activation=latent_activation
                )
        else:
            self.latent_dyn = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
        if self.latent_dyn is not None:
            for m in self.latent_dyn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _physics_step(self, state: Tensor, u: Tensor) -> Tensor:
        state_phys = state[..., : self.physical_size]
        if self.physical_size == 0 or self.physics_prior is None:
            return torch.zeros_like(state_phys)
        phys_dot = self.physics_prior(state_phys, u)
        return state_phys + self.dt * phys_dot

    def forward(self, input_t: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        B = input_t.shape[0]
        if hx is None:
            hx = torch.zeros(B, self.total_size, device=input_t.device, dtype=input_t.dtype)

        phys_next = self._physics_step(hx, input_t)

        if self.latent_size > 0:
            latent = hx[..., self.physical_size:]
            latent_dot = self.latent_dyn(torch.cat([hx, input_t], dim=-1))
            latent_next = latent + self.dt * latent_dot
        else:
            latent_next = None

        if self.latent_mode == "gate":
            candidate = torch.cat([phys_next, latent_next], dim=-1) if self.latent_size > 0 else phys_next
            z = torch.sigmoid(self.gate(torch.cat([hx, input_t], dim=-1)))
            return z * candidate + (1.0 - z) * hx

        elif self.latent_mode == "residual":
            if self.latent_size > 0:
                phys_next = phys_next + latent_next[..., : self.physical_size]
                candidate = torch.cat([phys_next, hx[..., self.physical_size:]], dim=-1)
            else:
                candidate = phys_next
            z = torch.sigmoid(self.gate(torch.cat([hx, input_t], dim=-1)))
            return z * candidate + (1.0 - z) * hx

        raise ValueError(f"Unknown mode {self.latent_mode}")

# ============================================================
# PhyGRU layer (Standard Version)
# ============================================================
class PhyGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        physics_prior: Optional[Union[PhysicsFn, nn.Module]],
        *,
        output_size: int = 1,
        physical_size: Optional[int] = None,
        latent_size: int = 0,
        dt: float = 1.0,
        batch_first: bool = True,
        output_from: Literal["state0", "physical0", "full"] = "state0",
        latent_type: Literal["linear", "mlp"] = "linear",
        latent_activation: str = "tanh",
        latent_hidden_size: Optional[int] = None,
        latent_num_layers: int = 2,
        gate_bias: bool = True,
        latent_mode="gate",
        return_sequences: bool = True,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.output_from = output_from
        self.return_sequences = return_sequences
        self.output_size = output_size

        self.cell = PhyGRUCell(
            input_size=input_size,
            state_size=state_size,
            physics_prior=physics_prior,
            physical_size=physical_size,
            latent_size=latent_size,
            dt=dt,
            latent_type=latent_type,
            latent_activation=latent_activation,
            latent_hidden_size=latent_hidden_size,
            latent_num_layers=latent_num_layers,
            gate_bias=gate_bias,
            latent_mode=latent_mode,
        )

        if output_from == "full":
            self.out_proj = nn.Linear(self.cell.total_size, output_size)
        else:
            self.out_proj = None

    def _project_output(self, state: Tensor) -> Tensor:
        if self.output_from == "state0":
            return state[..., : self.output_size]
        if self.output_from == "physical0":
            return state[..., : self.cell.physical_size][..., : self.output_size]
        if self.output_from == "full":
            return self.out_proj(state)
        raise ValueError("Invalid output_from")

    def forward(self, input_seq: Tensor, hx: Optional[Tensor] = None):
        if not self.batch_first:
            input_seq = input_seq.transpose(0, 1)
        B, T, _ = input_seq.shape
        state = hx
        outputs = []
        for t in range(T):
            state = self.cell(input_seq[:, t], state)
            outputs.append(self._project_output(state).unsqueeze(1))
        output = torch.cat(outputs, dim=1)
        if not self.return_sequences:
            output = output[:, -1]
        if not self.batch_first and self.return_sequences:
            output = output.transpose(0, 1)
        return output, state

# ============================================================
# Standard VehiclePhyGRU (Direct Core Output)
# ============================================================
class VehiclePhyGRU(nn.Module):
    def __init__(self, prior, dt, input_scaler, output_scaler):
        super().__init__()
        self.prior = prior
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

        def wrapped_prior(state, u_norm):
            u_phys = self.input_scaler.inverse_transform(u_norm.unsqueeze(1)).squeeze(1)
            return self.prior(state, u_phys)

        self.core = PhyGRU(
            input_size=2,
            state_size=4,
            physics_prior=wrapped_prior,
            output_size=1,
            physical_size=2,
            latent_size=2,
            dt=dt,
            latent_type="mlp",
            latent_activation="elu",
            latent_hidden_size=64,
            latent_num_layers=3,
            output_from="state0",
            latent_mode="residual"
        )

    def forward(self, u):
        y_norm, states = self.core(u)
        return y_norm, states

# ============================================================
# Baseline Models & Training (Rest of Code)
# ============================================================
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.fc(y), None

class LinearBicyclePrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.mass = nn.Parameter(torch.tensor(1500.0))
        self.iz = nn.Parameter(torch.tensor(2250.0))
        self.cf = nn.Parameter(torch.tensor(70000.0))
        self.cr = nn.Parameter(torch.tensor(80000.0))
        self.lf = 1.2
        self.lr = 1.6

    def forward(self, state, u):
        vy, r, delta, vx = state[:, 0], state[:, 1], u[:, 0], u[:, 1]
        af, ar = delta - (vy + self.lf * r) / vx, -(vy - self.lr * r) / vx
        vy_dot = (self.cf * af + self.cr * ar) / self.mass - vx * r
        r_dot = (self.lf * self.cf * af - self.lr * self.cr * ar) / self.iz
        return torch.stack([vy_dot, r_dot], dim=-1)

    def ay(self, state, u):
        return (self.forward(state, u)[:, 0] + u[:, 1] * state[:, 1]).unsqueeze(-1)

class PhysicsModel(nn.Module):
    def __init__(self, prior, dt, in_s, out_s):
        super().__init__()
        self.prior, self.dt, self.in_s, self.out_s = prior, dt, in_s, out_s

    def forward(self, u):
        B, T, _ = u.shape
        state = torch.zeros(B, 2, device=u.device)
        outs = []
        for t in range(T):
            u_p = self.in_s.inverse_transform(u[:, t:t+1]).squeeze(1)
            state = state + self.dt * self.prior(state, u_p)
            ay_n = self.out_s.transform(self.prior.ay(state, u_p).unsqueeze(1)).squeeze(1)
            outs.append(ay_n.unsqueeze(1))
        return torch.cat(outs, dim=1), state

def train_with_val(model, u_t, y_t, u_v, y_v, epochs=500, lr=3e-3, patience=50):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    best_state = None
    wait = 0

    for e in range(epochs):
        model.train()
        opt.zero_grad()
        yhat, _ = model(u_t)
        loss = loss_fn(yhat, y_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            yv, _ = model(u_v)
            val_loss = loss_fn(yv, y_v).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if e % 20 == 0:
            print(f"Epoch {e:03d} | Train: {loss.item():.6f} | Val: {val_loss:.6f}")
        if wait >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_loss

def build_models(dt, in_s, out_s):
    gru = GRUModel()
    prior_phys = LinearBicyclePrior()
    prior_phygru = LinearBicyclePrior()
    phys = PhysicsModel(prior_phys, dt, in_s, out_s)
    phygru = VehiclePhyGRU(prior_phygru, dt, in_s, out_s)
    return {"GRU": gru, "Physics": phys, "PhyGRU": phygru}

def evaluate_models_on_tests(models, test_sets):
    per_model_mses = {name: [] for name in models.keys()}

    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            for ut, yt in test_sets:
                yh, _ = model(ut)
                per_model_mses[name].append(nn.MSELoss()(yh, yt).item())

    return per_model_mses

def run_sensitivity_study(traj_lengths, n_train=16, n_val=16, n_test=8, dt=0.05):
    results = {
        "GRU": [],
        "Physics": [],
        "PhyGRU": []
    }

    for L in traj_lengths:
        print("\n" + "=" * 80)
        print(f"Trajectory length = {L}")
        print("=" * 80)

        u_tr, y_tr = generate_dataset(n_train, L, dt, 0.0)
        u_v, y_v   = generate_dataset(n_val,   L, dt, 10.0)

        in_s, out_s = TensorScaler(), TensorScaler()
        in_s.fit(u_tr)
        out_s.fit(y_tr)

        u_tr, y_tr = in_s.transform(u_tr), out_s.transform(y_tr)
        u_v, y_v   = in_s.transform(u_v), out_s.transform(y_v)

        TEST_LEN = 1000

        test_raw = [
            generate_dataset(1, TEST_LEN, dt, 20.0 + (i * 10))
            for i in range(n_test)
        ]

        test_sets = [
            (in_s.transform(u), out_s.transform(y))
            for u, y in test_raw
        ]

        models = build_models(dt, in_s, out_s)

        print("\nLearnable parameters:")
        for name, m in models.items():
            print(f"{name:<15} | {count_parameters(m)}")

        for name, m in models.items():
            print(f"\n--- Training {name} ---")
            train_with_val(m, u_tr, y_tr, u_v, y_v)

        mse_dict = evaluate_models_on_tests(models, test_sets)

        for name in results.keys():
            results[name].append(mse_dict[name])

        print("\nTest MSEs:")
        for name in ["Physics", "PhyGRU", "GRU"]:
            vals = mse_dict[name]
            print(f"{name:<15} | " + " | ".join([f"{v:.6f}" for v in vals]))

    return results

def plot_mse_envelopes(traj_lengths, results):
    plt.figure(figsize=(11, 6))

    for name in ["Physics", "PhyGRU", "GRU"]:
        arr = np.array(results[name])
        mean_mse = arr.mean(axis=1)
        min_mse = arr.min(axis=1)
        max_mse = arr.max(axis=1)

        plt.semilogy(traj_lengths, mean_mse, marker='o', linewidth=2, label=f"{name} mean")
        plt.fill_between(traj_lengths, min_mse, max_mse, alpha=0.18)

    plt.xticks(traj_lengths)
    plt.xlabel("Trajectory length")
    plt.ylabel("Test MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dt = 0.05
    traj_lengths = [5, 25, 50, 100, 200, 250, 300, 500]

    results = run_sensitivity_study(traj_lengths, n_train=16, n_val=16, n_test=8, dt=dt)
    plot_mse_envelopes(traj_lengths, results)