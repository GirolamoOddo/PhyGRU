"""

------------------------------------------------------------
PhyGRU API — parameter guide
------------------------------------------------------------

PhyGRU(
    input_size,
    state_size,
    physics_prior,
    *,
    output_size=1,
    physical_size=None,
    latent_size=0,
    dt=1.0,
    batch_first=True,
    output_from="state0",
    latent_type="linear",
    latent_activation="tanh",
    latent_hidden_size=None,
    latent_num_layers=2,
    gate_bias=True,
    latent_mode="gate",
    return_sequences=True,
)

Parameters
----------
input_size : int
    Number of input features per time step.
    Example: for a scalar control input u(t), use input_size=1.

state_size : int
    Total state dimension carried by the recurrent module.
    This is the size of the hidden state vector.

physics_prior : callable, nn.Module, or None
    A function/module with signature:
        physics_prior(state_phys, u) -> state_dot
    where:
        state_phys has shape (B, physical_size)
        u has shape (B, input_size)
        state_dot has shape (B, physical_size)
    If physical_size=0, set physics_prior=None and the model becomes purely learned.

output_size : int, default=1
    Size of the output sequence at each time step.
    If output_from="state0", the first output_size elements of the state are returned.
    If output_from="full", a learned linear head maps the full state to output_size.

physical_size : int or None, default=None
    Number of state dimensions governed by the physics prior.
    If None, it defaults to state_size.
    Set physical_size=0 for a pure learned model with no physics prior.
    Typical use: physical_size=2 for [x, x_dot].

latent_size : int, default=0
    Number of extra learned latent state dimensions.
    These are useful for unmodeled dynamics / correction terms.

dt : float, default=1.0
    Time step used in the explicit Euler integration inside the layer.

batch_first : bool, default=True
    If True, input shape is (B, T, F).
    If False, input shape is (T, B, F).

output_from : {"state0", "physical0", "full"}, default="state0"
    How outputs are produced:
    - "state0": return the first output_size components of the state
    - "physical0": return the first output_size components of the physical state
    - "full": project the full state through a learned linear layer

latent_type : {"linear", "mlp"}, default="linear"
    How latent dynamics are modeled when latent_size > 0.
    - "linear": one linear layer for latent dynamics
    - "mlp": multi-layer perceptron for latent dynamics

latent_activation : str, default="tanh"
    Activation used inside the latent MLP.
    Supported: tanh, relu, gelu, silu, elu, leaky_relu.

latent_hidden_size : int or None, default=None
    Hidden width of the latent MLP.
    If None, a reasonable default is used.

latent_num_layers : int, default=2
    Number of layers in the latent MLP.
    Use 1 for a single linear map (equivalent to latent_type="linear" behavior,
    but with the MLP wrapper still in place).

gate_bias : bool, default=True
    Whether to use bias in the update gate.

latent_mode : {"gate", "residual"}, default="gate"
    Latent can be applied or at gate level (as described in the original paper)
    or as a residual compensation on physical state.

return_sequences : bool, default=True
    If True, return the whole output sequence.
    If False, return only the last output.

Returns
-------
Forward call:
    output, h_n = model(input_seq, hx=None)

    output:
        If return_sequences=True:
            - batch_first=True  -> (B, T, output_size)
            - batch_first=False -> (T, B, output_size)
        Else:
            - (B, output_size)

    h_n:
        Final hidden state, shape (B, total_state_size)

------------------------------------------------------------

"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Reproducibility
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
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    if name in {"lrelu", "leaky_relu"}:
        return nn.LeakyReLU(0.01)
    raise ValueError(
        f"Unknown activation '{name}'. Supported: tanh, relu, gelu, silu, elu, leaky_relu."
    )


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "tanh",
        bias: bool = True,
    ) -> None:
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


@torch.no_grad()
def make_plots(title: str, x_true: Tensor, preds: dict[str, Tensor]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x_true[0, :, 0].cpu().numpy(), label="ground truth", linewidth=2)
    for name, y in preds.items():
        plt.plot(y[0, :, 0].cpu().numpy(), label=name, alpha=0.9)
    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel("normalized x")
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_model(
    model: nn.Module,
    u_train: Tensor,
    x_train: Tensor,
    u_val:   Tensor,
    x_val:   Tensor,
    *,
    epochs: int   = 150,
    patience: int = 50,
    lr: float     = 5e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    u_train = u_train.to(device)
    x_train = x_train.to(device)
    u_val = u_val.to(device)
    x_val = x_val.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        y_pred, _ = model(u_train)
        train_loss = loss_fn(y_pred, x_train)
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            y_val, _ = model(u_val)
            val_loss = loss_fn(y_val, x_val)

        tr = float(train_loss.item())
        va = float(val_loss.item())
        history["train"].append(tr)
        history["val"].append(va)

        if verbose and (epoch <= 5 or epoch % 25 == 0):
            print(f"epoch {epoch:03d} | train = {tr:.6e} | val = {va:.6e}")

        if va < best_val:
            best_val = va
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "model_state": best_state,
        "model": model,
    }


@torch.no_grad()
def evaluate(model: nn.Module, u: Tensor, x: Tensor):
    model.eval()
    y, _ = model(u)
    mse = nn.MSELoss()(y, x).item()
    return y, mse


# ============================================================
# PhyGRU cell
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
        latent_mode: Literal["gate", "residual"] = "gate",   # NEW
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        if state_size <= 0:
            raise ValueError("state_size must be > 0")
        if latent_size < 0:
            raise ValueError("latent_size must be >= 0")

        self.input_size = input_size
        self.state_size = state_size
        self.physical_size = physical_size if physical_size is not None else state_size
        self.latent_size = latent_size
        self.total_size = self.physical_size + self.latent_size
        self.dt = float(dt)
        self.physics_prior = physics_prior
        self.latent_mode = latent_mode   # NEW

        if self.physical_size > self.state_size:
            raise ValueError("physical_size cannot be larger than state_size")
        if self.physical_size == 0 and self.latent_size == 0:
            raise ValueError("At least one of physical_size or latent_size must be > 0")

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
                    activation=latent_activation,
                )
            else:
                raise ValueError("latent_type must be 'linear' or 'mlp'")
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
        device = input_t.device
        dtype = input_t.dtype

        if hx is None:
            hx = torch.zeros(B, self.total_size, device=device, dtype=dtype)

        # -----------------------------
        # physics
        # -----------------------------
        phys_next = self._physics_step(hx, input_t)

        # -----------------------------
        # latent dynamics
        # -----------------------------
        if self.latent_size > 0:
            latent = hx[..., self.physical_size :]
            latent_dot = self.latent_dyn(torch.cat([hx, input_t], dim=-1))
            latent_next = latent + self.dt * latent_dot
        else:
            latent_next = None

        # =====================================================
        # MODE 1 — ORIGINAL (latent inside GRU candidate)
        # =====================================================
        if self.latent_mode == "gate":

            if self.latent_size > 0:
                candidate = torch.cat([phys_next, latent_next], dim=-1)
            else:
                candidate = phys_next

            z = torch.sigmoid(self.gate(torch.cat([hx, input_t], dim=-1)))
            next_state = z * candidate + (1.0 - z) * hx
            return next_state

        # =====================================================
        # MODE 2 — RESIDUAL (latent corrects physics ONLY)
        # =====================================================
        elif self.latent_mode == "residual":

            # latent corrects ONLY physical state
            if self.latent_size > 0:
                phys_next = phys_next + latent_next[..., : self.physical_size]

            # latent NOT inside gate
            if self.latent_size > 0:
                candidate = torch.cat(
                    [phys_next, hx[..., self.physical_size:]], dim=-1
                )
            else:
                candidate = phys_next

            z = torch.sigmoid(self.gate(torch.cat([hx, input_t], dim=-1)))
            next_state = z * candidate + (1.0 - z) * hx
            return next_state

        else:
            raise ValueError(f"Unknown latent_mode '{self.latent_mode}'")

# ============================================================
# PhyGRU layer
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

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.batch_first = batch_first
        self.output_from = output_from
        self.return_sequences = return_sequences

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

        self.physical_size = self.cell.physical_size
        self.latent_size = latent_size
        self.total_size = self.cell.total_size

        if output_from == "full":
            self.out_proj = nn.Linear(self.total_size, output_size)
        else:
            self.out_proj = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.out_proj is not None:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def _project_output(self, state: Tensor) -> Tensor:
        if self.output_from == "state0":
            # first components of full state
            return state[..., : self.output_size]

        if self.output_from == "physical0":
            # slice ONLY from physical state
            state_phys = state[..., : self.physical_size]
            return state_phys[..., : self.output_size]

        if self.output_from == "full":
            return self.out_proj(state)

        raise ValueError(f"Unknown output_from='{self.output_from}'")

    def forward(self, input_seq: Tensor, hx: Optional[Tensor] = None):
        if input_seq.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(input_seq.shape)}")

        if not self.batch_first:
            input_seq = input_seq.transpose(0, 1)

        B, T, F = input_seq.shape
        if F != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {F}")

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
# Controlled system used as demo data
# ============================================================
def make_control_signal(T: int, seed_shift: float = 0.0) -> Callable[[int], float]:
    def u_fn(t: int) -> float:
        tt = float(t) + seed_shift
        return math.tanh(
            (0.35 - 0.00005 * tt)
            * (
                  (0.25 - 0.001 * tt) * math.sin((0.00007 + 0.0000001 * tt) * tt)
                + (0.10 + 0.001 * tt) * math.sin((0.000001 - 0.000001 * tt) * tt)
            )
        )
    return u_fn


def generate_system_dataset(
    T: int,
    dt: float,
    u_fn: Callable[[int], float],
):
    """
    True system slightly more complex than a plain spring-damper:
        x_ddot = u - 0.5*x_dot - 0.2*x + 0.12*tanh(x*x_dot) - 0.03*x^3
    """
    x, xd = 0.0, 0.0
    xs, us = [], []
    for t in range(T):
        u = float(u_fn(t))
        xdd = u - 0.5 * xd - 0.2 * x + 0.12 * math.tanh(x * xd) - 0.03 * (x ** 3)
        xd += dt * xdd
        x  += dt * xd
        xs.append([x])
        us.append([u])
    x = torch.tensor(xs, dtype=torch.float32).unsqueeze(0)
    u = torch.tensor(us, dtype=torch.float32).unsqueeze(0)
    return x, u


def split_sequence(x: Tensor, u: Tensor, n_train: int, n_val: int, n_test: int):
    total = x.shape[1]
    if n_train + n_val + n_test != total:
        raise ValueError("Split sizes must sum to total sequence length")
    x_train = x[:, :n_train]
    u_train = u[:, :n_train]
    x_val   = x[:, n_train : n_train + n_val]
    u_val   = u[:, n_train : n_train + n_val]
    x_test  = x[:, n_train + n_val :]
    u_test  = u[:, n_train + n_val :]
    return x_train, u_train, x_val, u_val, x_test, u_test


# ============================================================
# Main 
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dt       = 0.05
    T        = 3000
    epochs   = 200
    lr       = 1e-2
    patience = 20

    u_train_fn = make_control_signal(T, seed_shift=  0.0)
    u_val_fn   = make_control_signal(T, seed_shift=100.0)
    u_test_fn  = make_control_signal(T, seed_shift=200.0)

    x_train_full, u_train_full = generate_system_dataset(T=T, dt=dt, u_fn=u_train_fn)
    x_val_full,   u_val_full   = generate_system_dataset(T=T, dt=dt, u_fn=u_val_fn  )
    x_test_full,  u_test_full  = generate_system_dataset(T=T, dt=dt, u_fn=u_test_fn )

    # Normalization from train only, applied to all splits
    x_scale = x_train_full.abs().max().clamp_min(1e-12)
    u_scale = u_train_full.abs().max().clamp_min(1e-12)

    x_train_full = x_train_full / x_scale
    x_val_full   = x_val_full   / x_scale
    x_test_full  = x_test_full  / x_scale

    u_train_full = u_train_full / u_scale
    u_val_full   = u_val_full   / u_scale
    u_test_full  = u_test_full  / u_scale

    # Split each full sequence into train / val / test segments 
    n_train = int(T * 0.7)
    n_val   = int(T * 0.1)
    n_test  = int(T * 0.2)

    x_train, u_train, x_val, u_val, x_test, u_test = split_sequence(
        x_train_full, u_train_full, n_train, n_val, n_test
    )
    
    x_val, u_val, _, _, _, _   = split_sequence(x_val_full,  u_val_full,  n_train, n_val, n_test)
    x_test, u_test, _, _, _, _ = split_sequence(x_test_full, u_test_full, n_train, n_val, n_test)

    x_train_d = x_train.to(device)
    u_train_d = u_train.to(device)
    x_val_d   = x_val.to(device)
    u_val_d   = u_val.to(device)
    x_test_d  = x_test.to(device)
    u_test_d  = u_test.to(device)

    loss_fn = nn.MSELoss()

    print("\n========================================")
    print("PhyGRU progressive usage demo")
    print("========================================")
    print("All models are trained on the same system.")
    print("Train / val / test use different control actions, but the same plant.")
    print("Validation uses early stopping with patience.")
    print("Test MSE is reported for every model on the held-out test control action.\n")

    # --------------------------------------------------------
    # Step 0.0
    # --------------------------------------------------------
    print("\n==============================")
    print("Step 0.0: no physics + simple MLP latent correction")
    print("==============================")
    print("Physics prior: None")
    print("Latent: MLP with a single layer")

    model0 = PhyGRU(
        input_size=1,
        state_size=2,
        physical_size=0,
        latent_size=1,
        physics_prior=None,
        output_size=1,
        dt=dt,
        batch_first=True,
        output_from="state0",
        latent_type="mlp",
        latent_activation="elu",
        latent_hidden_size=4,
        latent_num_layers=1,
        return_sequences=True,
    ).to(device)

    print("Parameters:", count_parameters(model0))
    info0 = train_model(
        model0,
        u_train_d,
        x_train_d,
        u_val_d,
        x_val_d,
        epochs=epochs,
        patience=patience,
        lr=lr,
        verbose=True,
    )
    y0_test, test_mse0 = evaluate(model0, u_test_d, x_test_d)
    print(f"Step 0.0 best val epoch: {info0['best_epoch']} | best val loss: {info0['best_val_loss']:.6e} | test MSE: {test_mse0:.6e}")

    # --------------------------------------------------------
    # Step 0.1
    # --------------------------------------------------------
    print("\n==============================")
    print("Step 0.1: physics, no latent correction")
    print("==============================")
    print("Physics prior: MassSpringDamperPrior(mass=1.0, damping=0.5, stiffness=0.2)")
    print("Latent: None")

    # ============================================================
    # Physics priors
    # ============================================================
    class PhysicsPriorBase(nn.Module):
        def forward(self, state_phys: Tensor, u: Tensor) -> Tensor:
            raise NotImplementedError


    class MassSpringDamperPrior(PhysicsPriorBase):
        """
        State convention:
            state_phys[..., 0] = x
            state_phys[..., 1] = x_dot

        Dynamics:
            x_dot = v
            v_dot = (u - damping*v - stiffness*x) / mass
        """

        def __init__(
            self,
            mass: float = 1.0,
            damping: float = 0.5,
            stiffness: float = 0.2,
            learn_mass: bool = False,
            learn_damping: bool = False,
            learn_stiffness: bool = False,
            eps: float = 1e-12,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.mass = nn.Parameter(torch.tensor(float(mass)), requires_grad=learn_mass)
            self.damping = nn.Parameter(torch.tensor(float(damping)), requires_grad=learn_damping)
            self.stiffness = nn.Parameter(torch.tensor(float(stiffness)), requires_grad=learn_stiffness)

        def forward(self, state_phys: Tensor, u: Tensor) -> Tensor:
            if state_phys.shape[-1] < 2:
                raise ValueError("MassSpringDamperPrior expects [x, x_dot].")

            x = state_phys[..., 0]
            xd = state_phys[..., 1]
            u0 = u[..., 0] if u.shape[-1] > 1 else u.squeeze(-1)
            xdd = (u0 - self.damping * xd - self.stiffness * x) / (self.mass + self.eps)
            return torch.stack([xd, xdd], dim=-1)

    prior1 = MassSpringDamperPrior(
        mass=1.0,
        damping=0.5,
        stiffness=0.2,
        learn_mass=True,
        learn_damping=True,
        learn_stiffness=True,
    )

    model01 = PhyGRU(
        input_size=1,
        state_size=2,
        physical_size=2,
        latent_size=0,
        physics_prior=prior1,
        output_size=1,
        dt=dt,
        batch_first=True,
        output_from="state0",
        latent_type="linear",
        return_sequences=True,
    ).to(device)

    print("Parameters:", count_parameters(model01))
    info01 = train_model(
        model01,
        u_train_d,
        x_train_d,
        u_val_d,
        x_val_d,
        epochs=epochs,
        patience=patience,
        lr=lr,
        verbose=True,
    )
    y01_test, test_mse01 = evaluate(model01, u_test_d, x_test_d)
    print(f"Step 0.1 best val epoch: {info01['best_epoch']} | best val loss: {info01['best_val_loss']:.6e} | test MSE: {test_mse01:.6e}")

    # --------------------------------------------------------
    # Step 0.2
    # --------------------------------------------------------

    print("\n==============================")
    print("Step 02: refined prior, no latent correction")
    print("==============================")
    print("Physics prior: RefinedSpringDamperPrior(mass=1.0, damping=0.5, stiffness=0.2, nonlin_gain=0.05)")
    print("Latent: None")

    class RefinedSpringDamperPrior(PhysicsPriorBase):
      """
      A prior closer to the true system, but not identical.

      Example structure:
          x_ddot = (u - c*x_dot - k*x - nonlinear_term) / m
      """

      def __init__(
          self,
          mass: float = 1.0,
          damping: float = 0.45,
          stiffness: float = 0.18,
          nonlin_gain: float = 0.10,
          learn_mass: bool = False,
          learn_damping: bool = False,
          learn_stiffness: bool = False,
          learn_nonlin_gain: bool = False,
          eps: float = 1e-12,
      ) -> None:
          super().__init__()
          self.eps = eps
          self.mass = nn.Parameter(torch.tensor(float(mass)), requires_grad=learn_mass)
          self.damping = nn.Parameter(torch.tensor(float(damping)), requires_grad=learn_damping)
          self.stiffness = nn.Parameter(torch.tensor(float(stiffness)), requires_grad=learn_stiffness)
          self.nonlin_gain = nn.Parameter(torch.tensor(float(nonlin_gain)), requires_grad=learn_nonlin_gain)

      def forward(self, state_phys: Tensor, u: Tensor) -> Tensor:
          x  = state_phys[..., 0]
          xd = state_phys[..., 1]
          u0 = u[..., 0] if u.shape[-1] > 1 else u.squeeze(-1)
          nonlinear = self.nonlin_gain * x**3
          xdd = (u0 - self.damping * xd - self.stiffness * x - nonlinear) / (self.mass + self.eps)
          return torch.stack([xd, xdd], dim=-1)

    refined_prior0 = RefinedSpringDamperPrior(
        mass=1.0,
        damping=0.5,
        stiffness=0.2,
        nonlin_gain=0.05,
        learn_mass=True,
        learn_damping=True,
        learn_stiffness=True,
        learn_nonlin_gain=True,
    )

    model02 = PhyGRU(
        input_size=1,
        state_size=2,
        physical_size=2,
        latent_size=0,
        physics_prior=refined_prior0,
        output_size=1,
        dt=dt,
        batch_first=True,
        output_from="state0",
        latent_type="linear",
        return_sequences=True,
    ).to(device)

    print("Parameters:", count_parameters(model02))
    info02 = train_model(
        model02,
        u_train_d,
        x_train_d,
        u_val_d,
        x_val_d,
        epochs=epochs,
        patience=patience,
        lr=lr,
        verbose=True,
    )
    y02_test, test_mse02 = evaluate(model02, u_test_d, x_test_d)
    print(f"Step 02 best val epoch: {info02['best_epoch']} | best val loss: {info02['best_val_loss']:.6e} | test MSE: {test_mse02:.6e}")

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------
    print("\n==============================")
    print("Step 1: simple prior + linear latent correction")
    print("==============================")
    print("Physics prior: MassSpringDamperPrior(mass=1.0, damping=0.5, stiffness=0.2)")
    print("Latent: linear")

    prior1 = MassSpringDamperPrior(
        mass=1.0,
        damping=0.5,
        stiffness=0.2,
        learn_mass=True,
        learn_damping=True,
        learn_stiffness=True,
    )

    model1 = PhyGRU(
        input_size=1,
        state_size=3,
        physical_size=2,
        latent_size=1,
        physics_prior=prior1,
        output_size=1,
        dt=dt,
        batch_first=True,
        output_from="state0",
        latent_type="linear",
        return_sequences=True,
    ).to(device)

    print("Parameters:", count_parameters(model1))
    info1 = train_model(
        model1,
        u_train_d,
        x_train_d,
        u_val_d,
        x_val_d,
        epochs=epochs,
        patience=patience,
        lr=lr,
        verbose=True,
    )
    y1_test, test_mse1 = evaluate(model1, u_test_d, x_test_d)
    print(f"Step 1 best val epoch: {info1['best_epoch']} | best val loss: {info1['best_val_loss']:.6e} | test MSE: {test_mse1:.6e}")

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------
    print("\n==============================")
    print("Step 2: simple prior + MLP latent correction (single layer)")
    print("==============================")
    print("Physics prior: same as Step 1")
    print("Latent: MLP with a single layer")

    prior2 = MassSpringDamperPrior(
        mass=1.0,
        damping=0.5,
        stiffness=0.2,
        learn_mass=True,
        learn_damping=True,
        learn_stiffness=True,
    )

    model2 = PhyGRU(
        input_size=1,
        state_size=3,
        physical_size=2,
        latent_size=1,
        physics_prior=prior2,
        output_size=1,
        dt=dt,
        batch_first=True,
        output_from="state0",
        latent_type="mlp",
        latent_activation="elu",
        latent_hidden_size=4,
        latent_num_layers=1,
        return_sequences=True,
    ).to(device)

    print("Parameters:", count_parameters(model2))
    info2 = train_model(
        model2,
        u_train_d,
        x_train_d,
        u_val_d,
        x_val_d,
        epochs=epochs,
        patience=patience,
        lr=lr,
        verbose=True,
    )
    y2_test, test_mse2 = evaluate(model2, u_test_d, x_test_d)
    print(f"Step 2 best val epoch: {info2['best_epoch']} | best val loss: {info2['best_val_loss']:.6e} | test MSE: {test_mse2:.6e}")

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------
    print("\n==============================")
    print("Step 3: simple prior + MLP residual latent correction (multilayer)")
    print("==============================")
    print("Physics prior: same as Step 1")
    print("Latent: deeper MLP")

    prior3 = MassSpringDamperPrior(
        mass=1.0,
        damping=0.5,
        stiffness=0.2,
        learn_mass=True,
        learn_damping=True,
        learn_stiffness=True,
    )

    model3 = PhyGRU(
        input_size=1,
        state_size=3,
        physical_size=2,
        latent_size=1,
        physics_prior=prior3,
        output_size=1,
        dt=dt,
        batch_first=True,
        output_from="state0",
        latent_type="mlp",
        latent_activation="elu",
        latent_hidden_size=4,
        latent_num_layers=3,
        latent_mode="residual",
        return_sequences=True,
    ).to(device)

    print("Parameters:", count_parameters(model3))
    info3 = train_model(
        model3,
        u_train_d,
        x_train_d,
        u_val_d,
        x_val_d,
        epochs=epochs,
        patience=patience,
        lr=lr,
        verbose=True,
    )
    y3_test, test_mse3 = evaluate(model3, u_test_d, x_test_d)
    print(f"Step 3 best val epoch: {info3['best_epoch']} | best val loss: {info3['best_val_loss']:.6e} | test MSE: {test_mse3:.6e}")

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------
    print("\n==============================")
    print("Step 4: refined prior + MLP latent correction (multilayer)")
    print("==============================")
    print("Physics prior: refined prior closer to the true system")
    print("Latent: deeper MLP")


    refined_prior = RefinedSpringDamperPrior(
        mass=1.0,
        damping=0.5,
        stiffness=0.2,
        nonlin_gain=0.05,
        learn_mass=True,
        learn_damping=True,
        learn_stiffness=True,
        learn_nonlin_gain=True,
    )

    model4 = PhyGRU(
        input_size=1,
        state_size=3,
        physical_size=2,
        latent_size=1,
        physics_prior=refined_prior,
        output_size=1,
        dt=dt,
        batch_first=True,
        output_from="state0",
        latent_type="mlp",
        latent_activation="elu",
        latent_hidden_size=4,
        latent_num_layers=3,
        return_sequences=True,
    ).to(device)

    print("Parameters:", count_parameters(model4))
    info4 = train_model(
        model4,
        u_train_d,
        x_train_d,
        u_val_d,
        x_val_d,
        epochs=epochs,
        patience=patience,
        lr=lr,
        verbose=True,
    )
    y4_test, test_mse4 = evaluate(model4, u_test_d, x_test_d)
    print(f"Step 4 best val epoch: {info4['best_epoch']} | best val loss: {info4['best_val_loss']:.6e} | test MSE: {test_mse4:.6e}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n========================================")
    print("Summary (test set)")
    print("========================================")
    print(f"Step 00 | no physics + simple MLP correction                     | Test MSE = {test_mse0:.6e}" )
    print(f"Step 01 | simple prior,   no latent                              | Test MSE = {test_mse01:.6e}")
    print(f"Step 02 | refined prior,  no latent                              | Test MSE = {test_mse02:.6e}")
    print(f"Step 1  | simple prior  + linear latent correction               | Test MSE = {test_mse1:.6e}" )
    print(f"Step 2  | simple prior  + MLP latent (single layer)              | Test MSE = {test_mse2:.6e}" )
    print(f"Step 3  | simple prior  + MLP residual latent (multilayer)       | Test MSE = {test_mse3:.6e}" )
    print(f"Step 4  | refined prior + MLP latent (multilayer)                | Test MSE = {test_mse4:.6e}" )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    make_plots(
        "Validation of progression on the held-out test control action",
        x_test_d,
        {
            "step 0":  y0_test.detach().cpu(),
            "step 01": y01_test.detach().cpu(),
            "step 02": y02_test.detach().cpu(),
            "step 1":  y1_test.detach().cpu(),
            "step 2":  y2_test.detach().cpu(),
            "step 3":  y3_test.detach().cpu(),
            "step 4":  y4_test.detach().cpu(),
        },
    )

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    ax[0].plot(x_test_d[0, :, 0].cpu().numpy(), label="ground truth", linewidth=2)
    ax[0].plot(y0_test[ 0, :, 0].detach().cpu().numpy(), label="step 00")
    ax[0].plot(y01_test[0, :, 0].detach().cpu().numpy(), label="step 01")
    ax[0].plot(y02_test[0, :, 0].detach().cpu().numpy(), label="step 02")
    ax[0].plot(y1_test[ 0, :, 0].detach().cpu().numpy(), label="step 1")
    ax[0].plot(y2_test[ 0, :, 0].detach().cpu().numpy(), label="step 2")
    ax[0].plot(y3_test[ 0, :, 0].detach().cpu().numpy(), label="step 3")
    ax[0].plot(y4_test[ 0, :, 0].detach().cpu().numpy(), label="step 4")
    ax[0].set_title("Test set comparison on the refined system")
    ax[0].legend()

    ax[1].plot(info0["history"]["train"], label="step 0 train")
    ax[1].plot(info0["history"]["val"], label="step 0 val")
    ax[1].plot(info4["history"]["train"], label="step 4 train")
    ax[1].plot(info4["history"]["val"], label="step 4 val")
    ax[1].set_title("Training/validation curves")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("MSE")
    ax[1].set_yscale("log")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


###############################################################################
## RESULTS

Device: cpu

========================================
PhyGRU progressive usage demo
========================================
All models are trained on the same system.
Train / val / test use different control actions, but the same plant.
Validation uses early stopping with patience.
Test MSE is reported for every model on the held-out test control action.


==============================
Step 0.0: no physics + simple MLP latent correction
==============================
Physics prior: None
Latent: MLP with a single layer
Parameters: 6
epoch 001 | train = 1.699562e-01 | val = 1.027578e-01
epoch 002 | train = 9.761158e-02 | val = 6.255624e-02
epoch 003 | train = 6.050610e-02 | val = 4.609314e-02
epoch 004 | train = 4.622190e-02 | val = 4.298883e-02
epoch 005 | train = 4.455793e-02 | val = 4.548773e-02
epoch 025 | train = 1.930033e-02 | val = 2.317217e-02
epoch 050 | train = 1.116941e-02 | val = 1.421644e-02
epoch 075 | train = 8.030207e-03 | val = 9.948278e-03
epoch 100 | train = 6.208704e-03 | val = 7.623192e-03
epoch 125 | train = 5.069218e-03 | val = 6.279157e-03
epoch 150 | train = 4.299025e-03 | val = 5.377324e-03
epoch 175 | train = 3.752060e-03 | val = 4.728810e-03
epoch 200 | train = 3.347690e-03 | val = 4.248887e-03
Step 0.0 best val epoch: 200 | best val loss: 4.248887e-03 | test MSE: 4.415157e-03

==============================
Step 0.1: physics, no latent correction
==============================
Physics prior: MassSpringDamperPrior(mass=1.0, damping=0.5, stiffness=0.2)
Latent: None
Parameters: 11
epoch 001 | train = 3.374838e+00 | val = 2.459637e+00
epoch 002 | train = 2.449113e+00 | val = 1.905900e+00
epoch 003 | train = 1.892531e+00 | val = 1.544130e+00
epoch 004 | train = 1.530629e+00 | val = 1.292639e+00
epoch 005 | train = 1.279855e+00 | val = 1.109226e+00
epoch 025 | train = 3.018898e-01 | val = 2.963467e-01
epoch 050 | train = 1.730328e-01 | val = 1.730166e-01
epoch 075 | train = 1.245036e-01 | val = 1.248922e-01
epoch 100 | train = 9.493203e-02 | val = 9.529608e-02
epoch 125 | train = 7.458612e-02 | val = 7.487532e-02
epoch 150 | train = 5.989385e-02 | val = 6.010808e-02
epoch 175 | train = 4.893180e-02 | val = 4.908165e-02
epoch 200 | train = 4.053923e-02 | val = 4.063706e-02
Step 0.1 best val epoch: 200 | best val loss: 4.063706e-02 | test MSE: 4.202327e-02

==============================
Step 02: refined prior, no latent correction
==============================
Physics prior: RefinedSpringDamperPrior(mass=1.0, damping=0.5, stiffness=0.2, nonlin_gain=0.05)
Latent: None
Parameters: 12
epoch 001 | train = 3.660528e-01 | val = 3.087804e-01
epoch 002 | train = 3.011255e-01 | val = 2.592160e-01
epoch 003 | train = 2.523118e-01 | val = 2.207647e-01
epoch 004 | train = 2.145519e-01 | val = 1.902457e-01
epoch 005 | train = 1.846643e-01 | val = 1.655671e-01
epoch 025 | train = 2.860340e-02 | val = 2.778133e-02
epoch 050 | train = 8.542629e-03 | val = 8.446121e-03
epoch 075 | train = 3.750366e-03 | val = 3.709291e-03
epoch 100 | train = 1.861792e-03 | val = 1.840190e-03
epoch 125 | train = 9.930808e-04 | val = 9.882108e-04
epoch 150 | train = 5.744971e-04 | val = 5.828769e-04
epoch 175 | train = 3.677458e-04 | val = 3.849838e-04
epoch 200 | train = 2.627916e-04 | val = 2.848578e-04
Step 02 best val epoch: 200 | best val loss: 2.848578e-04 | test MSE: 2.945200e-04

==============================
Step 1: simple prior + linear latent correction
==============================
Physics prior: MassSpringDamperPrior(mass=1.0, damping=0.5, stiffness=0.2)
Latent: linear
Parameters: 23
epoch 001 | train = 1.396484e+00 | val = 1.278358e+00
epoch 002 | train = 1.502521e+00 | val = 1.372631e+00
epoch 003 | train = 1.280237e+00 | val = 1.279528e+00
epoch 004 | train = 1.051014e+00 | val = 1.155916e+00
epoch 005 | train = 7.701920e-01 | val = 9.578959e-01
epoch 025 | train = 1.678043e-01 | val = 1.685748e-01
Early stopping at epoch 33 (best val epoch 13)
Step 1 best val epoch: 13 | best val loss: 1.530095e-01 | test MSE: 1.972263e-01

==============================
Step 2: simple prior + MLP latent correction (single layer)
==============================
Physics prior: same as Step 1
Latent: MLP with a single layer
Parameters: 23
epoch 001 | train = 1.385128e+00 | val = 1.256130e+00
epoch 002 | train = 1.209035e+00 | val = 1.101026e+00
epoch 003 | train = 1.060307e+00 | val = 9.696071e-01
epoch 004 | train = 9.343099e-01 | val = 8.578805e-01
epoch 005 | train = 8.272057e-01 | val = 7.625470e-01
epoch 025 | train = 1.625883e-01 | val = 1.561951e-01
epoch 050 | train = 6.817893e-02 | val = 6.764874e-02
Early stopping at epoch 58 (best val epoch 38)
Step 2 best val epoch: 38 | best val loss: 5.329328e-02 | test MSE: 6.632829e-02

==============================
Step 3: simple prior + MLP residual latent correction (multilayer)
==============================
Physics prior: same as Step 1
Latent: deeper MLP
Parameters: 63
epoch 001 | train = 1.626224e-01 | val = 2.266335e-02
epoch 002 | train = 2.303752e-02 | val = 5.493690e-02
epoch 003 | train = 5.630982e-02 | val = 7.131889e-02
epoch 004 | train = 7.280517e-02 | val = 7.081323e-02
epoch 005 | train = 7.186949e-02 | val = 6.116526e-02
epoch 025 | train = 5.713805e-03 | val = 8.083289e-03
epoch 050 | train = 2.648383e-03 | val = 2.843067e-03
epoch 075 | train = 2.244880e-03 | val = 2.486506e-03
epoch 100 | train = 2.014219e-03 | val = 2.222688e-03
epoch 125 | train = 1.822395e-03 | val = 1.949540e-03
epoch 150 | train = 1.652660e-03 | val = 1.729347e-03
epoch 175 | train = 1.500962e-03 | val = 1.536382e-03
epoch 200 | train = 1.364488e-03 | val = 1.375018e-03
Step 3 best val epoch: 200 | best val loss: 1.375018e-03 | test MSE: 1.382537e-03

==============================
Step 4: refined prior + MLP latent correction (multilayer)
==============================
Physics prior: refined prior closer to the true system
Latent: deeper MLP
Parameters: 64
epoch 001 | train = 3.661473e-01 | val = 6.165175e-01
epoch 002 | train = 1.562022e-01 | val = 1.347727e-01
epoch 003 | train = 7.322700e-02 | val = 1.054156e-01
epoch 004 | train = 1.069108e-01 | val = 1.119109e-01
epoch 005 | train = 1.272071e-01 | val = 1.194793e-01
epoch 025 | train = 6.666347e-02 | val = 1.245006e-01
Early stopping at epoch 36 (best val epoch 16)
Step 4 best val epoch: 16 | best val loss: 1.053316e-01 | test MSE: 2.047191e-01

========================================
Summary (test set)
========================================
Step 00 | no physics + simple MLP correction                     | Test MSE = 4.415157e-03
Step 01 | simple prior,   no latent                              | Test MSE = 4.202327e-02
Step 02 | refined prior,  no latent                              | Test MSE = 2.945200e-04
Step 1  | simple prior  + linear latent correction               | Test MSE = 1.972263e-01
Step 2  | simple prior  + MLP latent (single layer)              | Test MSE = 6.632829e-02
Step 3  | simple prior  + MLP residual latent (multilayer)       | Test MSE = 1.382537e-03
Step 4  | refined prior + MLP latent (multilayer)                | Test MSE = 2.047191e-01


