# ============================================================
# PhyGRU (with bias flag) + Minimal training / save / reload / inference example
# Single file, minimal and runnable as a user guide
# ============================================================

from typing import Optional, Callable, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Activation utility
# -----------------------------
def _get_activation(act: Union[str, Callable, None]) -> Callable:
    if act is None or act == 'identity':
        return nn.Identity()
    act = act.lower() if isinstance(act, str) else act
    if act == 'tanh':
        return nn.Tanh()
    if act == 'relu':
        return nn.ReLU()
    if act == 'gelu':
        return nn.GELU()
    if callable(act):
        return act
    raise ValueError(f"Unsupported activation: {act}")

# -----------------------------
# PhyGRUCell / PhyGRU (bias flag included)
# -----------------------------
class PhyGRUCell(nn.Module):
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        physics_law: nn.Module,
        latent_dim: int = 0,
        latent_activation: Union[str, Callable, None] = 'tanh',
        use_reset: bool = True,
        dt: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.physics_law = physics_law
        self.dt = float(dt)
        self.total_state = self.state_dim + self.latent_dim

        if self.latent_dim > 0:
            self.latent_dyn = nn.Linear(
                self.state_dim + self.latent_dim + self.input_dim,
                self.latent_dim,
                bias=bias
            )
            self.latent_activation = _get_activation(latent_activation)
            if use_reset:
                self.r_gate = nn.Sequential(
                    nn.Linear(
                        self.state_dim + self.latent_dim + self.input_dim,
                        self.latent_dim,
                        bias=bias
                    ),
                    nn.Sigmoid()
                )
            else:
                self.r_gate = None
        else:
            self.latent_dyn = None
            self.latent_activation = nn.Identity()
            self.r_gate = None

        self.z_gate = nn.Sequential(
            nn.Linear(
                self.total_state + self.input_dim,
                self.total_state,
                bias=bias
            ),
            nn.Sigmoid()
        )

    def forward(self, u: torch.Tensor, hx: Optional[torch.Tensor]) -> torch.Tensor:
        B = u.shape[0]
        if hx is None:
            hx = torch.zeros(B, self.total_state, dtype=u.dtype, device=u.device)

        phys = hx[:, :self.state_dim]
        latent = hx[:, self.state_dim:] if self.latent_dim > 0 else None

        phys_dot = self.physics_law(phys, u)
        phys_next = phys + self.dt * phys_dot

        if self.latent_dim > 0:
            if self.r_gate is not None:
                r = self.r_gate(torch.cat([phys, latent, u], dim=1))
                gated_latent = r * latent
            else:
                gated_latent = latent

            latent_input = torch.cat([phys, gated_latent, u], dim=1)
            latent_dot = self.latent_dyn(latent_input)
            latent_dot = self.latent_activation(latent_dot)
            latent_next = latent + self.dt * latent_dot
            candidate = torch.cat([phys_next, latent_next], dim=1)
        else:
            candidate = phys_next

        z = self.z_gate(torch.cat([hx, u], dim=1))
        next_state = z * candidate + (1.0 - z) * hx
        return next_state

class PhyGRU(nn.Module):
    def __init__(
        self,
        physics_law: nn.Module,
        state_dim: int,
        input_dim: int,
        latent_dim: int = 0,
        latent_activation: Union[str, Callable, None] = 'identity',
        use_reset: bool = True,
        dt: float = 1.0,
        batch_first: bool = True,
        output: str = 'full',
        bias: bool = True,
    ):
        super().__init__()
        assert output in ('physical', 'full'), "output must be 'physical' or 'full'"
        self.cell = PhyGRUCell(
            state_dim, input_dim, physics_law,
            latent_dim=latent_dim,
            latent_activation=latent_activation,
            use_reset=use_reset,
            dt=dt,
            bias=bias
        )
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.total_state = state_dim + latent_dim
        self.input_dim = input_dim
        self.batch_first = batch_first
        self.output = output

    def forward(self,
                inputs: torch.Tensor,
                hx: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.batch_first:
            inputs = inputs.permute(1, 0, 2)

        B, T, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        if hx is None:
            hx = torch.zeros(B, self.total_state, device=device, dtype=dtype)
        else:
            if hx.shape != (B, self.total_state):
                raise ValueError(f"hx must have shape (B, {self.total_state}), got {hx.shape}")

        outputs = []
        state = hx
        for t in range(T):
            u_t = inputs[:, t, :]
            state = self.cell(u_t, state)
            if self.output == 'full':
                out_t = state
            else:
                out_t = state[:, :self.state_dim]
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)
        last_state = state
        return outputs, last_state



################################################################################
################################################################################
# -----------------------------
# Minimal full pipeline: create synthetic data, train, save, reload, infer
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # settings
    BATCH = 16
    N_SAMPLES = 256
    EPOCHS = 20
    LR = 1e-3
    T = 10
    input_dim = 1
    phys_state_dim = 2 ## must be coherent with declared physics law
    phys_latent_dim = 3
    gru_hidden_dim = 32
    output_dim = 1
    dt = 0.01
    device = torch.device("cpu")

    # -----------------------------
    # Example physics law provided to PhyGRU
    # -----------------------------
    class SecondOrderLaw(nn.Module):
        def __init__(self, learn_a=True, learn_b=True, learn_c=True):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(0.5), requires_grad=learn_a)
            self.b = nn.Parameter(torch.tensor(0.6), requires_grad=learn_b)
            self.c = nn.Parameter(torch.tensor(0.7), requires_grad=learn_c)

        def forward(self, state, u):
            x   = state[:, 0]
            xd  = state[:, 1]
            u_s = u.squeeze(-1) if u.dim() > 1 else u
            xdd = (u_s - self.b * xd - self.c * x) / (self.a + 1e-12)
            return torch.stack([xd, xdd], dim=1)

    # -----------------------------
    # Simple model: PhyGRU -> GRU -> FNN
    # -----------------------------
    class SimpleHybridModel(nn.Module):
        def __init__(
            self,
            physics_law,
            input_dim,
            phys_state_dim,
            phys_latent_dim,
            gru_hidden_dim,
            output_dim,
            dt=0.01,
        ):
            super().__init__()

            self.phygru = PhyGRU(
                physics_law=physics_law,
                state_dim=phys_state_dim,
                input_dim=input_dim,
                latent_dim=phys_latent_dim,
                latent_activation='identity',
                use_reset=False,
                dt=dt,
                batch_first=True,
                output='full',   # <--- ensure downstream GRU input matches (*)
                bias=False
            )

            self.gru = nn.GRU(
                input_size=phys_state_dim+phys_latent_dim, # <--- (*)
                hidden_size=gru_hidden_dim,
                batch_first=True
            )

            self.fnn = nn.Sequential(
                nn.Linear(gru_hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )

        def forward(self, u_seq):
            x, _ = self.phygru(u_seq)     # (B, T, phys_state_dim)
            x, _ = self.gru(x)            # (B, T, gru_hidden_dim)
            y = self.fnn(x)               # (B, T, output_dim)
            return y


    ############################################################################
    ############################################################################
    # Dummy dataset Generation
    
    # Ground-truth physics used to generate synthetic targets (known params)
    physics_gt = SecondOrderLaw(learn_a=False, learn_b=False, learn_c=False)
    # set known ground-truth parameters for simulation
    with torch.no_grad():
        physics_gt.a.copy_(torch.tensor(1.0))
        physics_gt.b.copy_(torch.tensor(0.5))
        physics_gt.c.copy_(torch.tensor(0.2))

    # Generate synthetic dataset:
    # For each sample: random u_seq, simulate physics to get x(t) (first state component)
    def simulate_physics_batch(u_seq, physics_module, dt):
        # u_seq: (B, T, 1)
        B, T, _ = u_seq.shape
        state = torch.zeros(B, phys_state_dim)
        traj = []
        for t in range(T):
            u_t = u_seq[:, t, :]
            state = state + dt * physics_module(state, u_t)  # Euler
            traj.append(state[:, 0:1])  # record first physical variable as target (B,1)
        traj = torch.stack(traj, dim=1) # (B, T, 1)
        return traj

    # create all samples (N_SAMPLES x T x 1)
    all_u = torch.randn(N_SAMPLES, T, input_dim) * 2.0  
    all_y = simulate_physics_batch(all_u, physics_gt, dt)  # (N_SAMPLES, T, 1)

    dataset = TensorDataset(all_u, all_y)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    ############################################################################

    # instantiate model
    model = SimpleHybridModel(
        physics_law=SecondOrderLaw(),  # physics inside model 
        input_dim=input_dim,
        phys_state_dim=phys_state_dim,
        phys_latent_dim=phys_latent_dim,
        gru_hidden_dim=gru_hidden_dim,
        output_dim=output_dim,
        dt=dt
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # training loop 
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)  # (B, T, output_dim)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch}/{EPOCHS}    Loss: {epoch_loss:.6f}")

    # save model (state_dict)
    save_path = "phygru_hybrid_example.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_args": {
            "input_dim": input_dim,
            "phys_state_dim": phys_state_dim,
            "phys_latent_dim": phys_latent_dim,
            "gru_hidden_dim": gru_hidden_dim,
            "output_dim": output_dim,
            "dt": dt
        }
    }, save_path)
    print(f"Model saved to {save_path}")

    # ====== reload model for inference ======
    ckpt = torch.load(save_path, map_location=device)
    args = ckpt["model_args"]

    model2 = SimpleHybridModel(
        physics_law=SecondOrderLaw(),
        input_dim=args["input_dim"],
        phys_state_dim=args["phys_state_dim"],
        phys_latent_dim=args["phys_latent_dim"],
        gru_hidden_dim=args["gru_hidden_dim"],
        output_dim=args["output_dim"],
        dt=args["dt"]
    ).to(device)
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.eval()

    # inference on a small test batch (take first 8 samples)
    test_u = all_u[:8].to(device)
    test_y_true = all_y[:8].to(device)
    with torch.no_grad():
        test_pred = model2(test_u)

    print("Test pred shape:", test_pred.shape)
    print("Test true shape:", test_y_true.shape)
    print("First-sample first-timestep true vs pred:",
          test_y_true[0, 0].item(), "vs", test_pred[0, 0, 0].item())
