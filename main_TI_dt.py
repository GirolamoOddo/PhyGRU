
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import random
import numpy as np
import os
import sys

print("PhyGRU dt experiment script (data_dt fixed at 0.01)")
print(f"Python: {sys.version.split()[0]} | torch: {torch.__version__} | numpy: {np.__version__}")

# reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Global simulation parameters (data sequence length in samples)
T = 6000

# Utilities

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inference_time(model, u):
    with torch.no_grad():
        t0 = time.time()
        _ = model(u)
        t1 = time.time()
    B, TT, _ = u.shape
    return (t1 - t0) / (B * TT)

# Spearman as in baseline
import math

def _rankdata(a):
    a = np.asarray(a)
    n = a.shape[0]
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty(n, dtype=int)
    inv[sorter] = np.arange(n)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and a[sorter[j]] == a[sorter[i]]:
            j += 1
        rank = 0.5 * (i + 1 + j)
        ranks[sorter[i:j]] = rank
        i = j
    return ranks


def spearmanr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size:
        raise ValueError("Input arrays must have the same length for Spearman correlation.")
    rx = _rankdata(a)
    ry = _rankdata(b)
    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    num = np.sum(rxm * rym)
    den = math.sqrt(np.sum(rxm**2) * np.sum(rym**2))
    if den == 0:
        return 0.0
    return num / den

# Dataset generators - always use provided local_dt (no global dt)

def generate_data_1(u_fn, local_dt, T_local=None):
    if T_local is None:
        T_local = T
    x, xd = 0.0, 0.0
    xs, us = [], []
    for t in range(T_local):
        u = u_fn(t)
        xdd = (u - 0.5 * xd - 0.2 * x)
        xd += local_dt * xdd
        x  += local_dt * xd
        xs.append([x])
        us.append([u])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(us, dtype=torch.float32)

# (Other systems kept for completeness but follow same signature)

def generate_data_2(u_fn, local_dt, T_local=None):
    if T_local is None:
        T_local = T
    x, xd = 0.0, 0.0
    xs, us = [], []
    for t in range(T_local):
        u = u_fn(t)
        xdd = (u - 0.5 * xd - 0.2 * x + math.tanh(x*xd))
        xd += local_dt * xdd
        x  += local_dt * xd
        xs.append([x])
        us.append([u])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(us, dtype=torch.float32)


def generate_data_3(u_fn, local_dt, T_local=None):
    if T_local is None:
        T_local = T
    x, xd, xdd = 0.0, 0.0, 0.0
    xs, us = [], []
    for t in range(T_local):
        u = u_fn(t)
        xddd = u - 3 * xdd - 2*math.tanh(xd) - 0.1 * math.tanh(x)
        xdd += local_dt * xddd
        xd  += local_dt * xdd
        x   += local_dt * xd
        xs.append([x])
        us.append([u])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(us, dtype=torch.float32)

# PhyGRU implementation (model keeps its own integration timestep `model_dt`)
class PhyGRUCell(nn.Module):
    def __init__(self, state_dim, input_dim, physics_law, latent_dim=0, model_dt=0.01):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.physics_law = physics_law
        self.model_dt = model_dt
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
        # physics law expects state[:, :state_dim], u is (B,input_dim)
        phys_dot = self.physics_law(state[:, :self.state_dim], u)
        # use self.model_dt (this is the deliberate mismatch parameter)
        phys_next = state[:, :self.state_dim] + self.model_dt * phys_dot
        if self.latent_dim > 0:
            latent = state[:, self.state_dim:]
            latent_dot = self.latent_dyn(torch.cat([state, u], dim=1))
            latent_next = latent + self.model_dt * latent_dot
            candidate = torch.cat([phys_next, latent_next], dim=1)
        else:
            candidate = phys_next
        z = self.z_gate(torch.cat([state, u], dim=1))
        return z * candidate + (1 - z) * state

class PhyGRU(nn.Module):
    def __init__(self, physics_law, state_dim, input_dim, latent_dim=0, model_dt=0.01):
        super().__init__()
        self.cell = PhyGRUCell(state_dim, input_dim, physics_law, latent_dim, model_dt=model_dt)
        self.state_dim = state_dim
        self.latent_dim = latent_dim

    def set_model_dt(self, new_dt):
        self.cell.model_dt = new_dt

    def forward(self, u_seq):
        B, Tt, _ = u_seq.shape
        state = torch.zeros(B, self.state_dim + self.latent_dim, dtype=u_seq.dtype, device=u_seq.device)
        ys = []
        for t in range(Tt):
            state = self.cell(state, u_seq[:, t])
            ys.append(state[:, 0:1])
        return torch.stack(ys, dim=1)

# Physics prior (Mass-Spring-Damper)
class MassSpringDamperLaw(nn.Module):
    def __init__(self, learn_a=True, learn_b=True, learn_c=True):
        super().__init__()
        # initial values kept as in baseline
        self.a = nn.Parameter(torch.tensor(0.5), requires_grad=learn_a)
        self.b = nn.Parameter(torch.tensor(0.6), requires_grad=learn_b)
        self.c = nn.Parameter(torch.tensor(0.7), requires_grad=learn_c)

    def forward(self, state, u):
        # state: tensor with columns [x, xd]
        x, xd = state[:, 0], state[:, 1]
        u_s = u.squeeze()
        xdd = (u_s - self.b * xd - self.c * x) / (self.a + 1e-12)
        return torch.stack([xd, xdd], dim=1)

# Training function (kept same as baseline but simplified for PhyGRU-only runs)
def train_with_validation(model, u_train, x_train, u_val, x_val, save_path, epochs=100, lr=1e-2, verbose=False):
    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    model.to(device)
    u_train = u_train.to(device)
    x_train = x_train.to(device)
    u_val   = u_val.to(device)
    x_val   = x_val.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        y_train_pred = model(u_train)
        train_loss = loss_fn(y_train_pred, x_train)

        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            y_val = model(u_val)
            val_loss = loss_fn(y_val, x_val).item()

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"  [epoch {epoch}] New best val loss: {best_val:.3e} -> saved to {save_path}")

        if verbose and (epoch % 50 == 0 or epoch <= 5):
            print(f"  epoch {epoch:03d} | train_loss={train_loss.item():.3e} | val_loss={val_loss:.3e}")

    if best_epoch >= 0:
        model.load_state_dict(torch.load(save_path))

    return {
        "best_val_loss": best_val,
        "best_epoch":    best_epoch,
        "history":       history,
        "model_path":    save_path
    }

# ---------------------------
# Analytical DT_critical computation for explicit Euler stability
# ---------------------------
def compute_dt_critical(a, b, c):
    alpha = -b / (2.0 * a)
    disc = 4.0 * c / a - (b / a) ** 2

    if disc > 0:   # oscillatory case
        omega = math.sqrt(disc) / 2.0
    else:
        omega = 0.0

    denom = alpha * alpha + omega * omega
    if denom <= 0:
        return float('inf')

    dt_crit = -2.0 * alpha / denom
    return dt_crit


# Experiment runner: train PhyGRU (latent=1) with internal model_dt=train_dt
# All sequences (train/val/test) are generated at data_dt=0.01. No resampling is applied.

def run_phygru_dt_experiments(train_dt, test_dts, results_dir='results_npz', checkpoints_dir='checkpoints', EPOCHS=150, LR=5e-3):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    data_dt = 0.01   # ALWAYS generate physics trajectories with dt=0.01
    print(f"\n--- Running experiment: model_dt (train) = {train_dt} | test_dts = {test_dts} | data_dt = {data_dt} ---")

    # Data generation for Sys_1 at data_dt
    x_train, u_train = generate_data_1(
        lambda t: math.tanh((0.3-0.00005*t)*((0.25-0.001*t)*math.sin((0.00007+0.0000001*t)*t) + (0.10+0.001*t)*math.sin((0.000001-0.000001*t)*t))),
        local_dt=data_dt
    )
    x_train = x_train.unsqueeze(0)
    u_train = u_train.unsqueeze(0)

    x_val, u_val = generate_data_1(
        lambda t: math.tanh((0.4-0.00005*t)*((0.35-0.003*t)*math.sin((0.00009+0.0000002*t)*t) + (0.15+0.001*t)*math.sin((0.000003-0.000001*t)*t))),
        local_dt=data_dt
    )
    x_val = x_val.unsqueeze(0)
    u_val = u_val.unsqueeze(0)

    # Normalization (train-only scalers computed from data generated at data_dt)
    x_abs_max = float(torch.max(torch.abs(x_train)).item())
    if x_abs_max == 0.0: x_abs_max = 1.0
    u_abs_max = float(torch.max(torch.abs(u_train)).item())
    if u_abs_max == 0.0: u_abs_max = 1.0
    x_train = x_train / x_abs_max
    x_val   = x_val   / x_abs_max
    u_train = u_train / u_abs_max
    u_val   = u_val   / u_abs_max

    # Build PhyGRU model (latent=1) with internal model_dt = train_dt
    key = f"PhyGRU_lat1_modeldt{train_dt}"
    print(f"Training {key} | params = {count_parameters(PhyGRU(MassSpringDamperLaw(), 2, 1, latent_dim=1, model_dt=train_dt))}")

    model = PhyGRU(MassSpringDamperLaw(), 2, 1, latent_dim=1, model_dt=train_dt)
    ckpt_path = os.path.join(checkpoints_dir, f"Sys_1_{key}_best.pt")

    info = train_with_validation(
        model, u_train, x_train, u_val, x_val,
        save_path=ckpt_path, epochs=EPOCHS, lr=LR, verbose=True
    )

    # Load best model
    best_model = PhyGRU(MassSpringDamperLaw(), 2, 1, latent_dim=1, model_dt=train_dt)
    best_model.load_state_dict(torch.load(ckpt_path))
    best_model.eval()

    # For each test_dt evaluate model's behavior while keeping test sequences generated at data_dt
    results = {}
    for test_dt in test_dts:
        print(f"\nEvaluating checkpoint trained at model_dt={train_dt} on model_dt={test_dt} (data_dt={data_dt})")
        x_test, u_test = generate_data_1(
            lambda t: math.tanh((0.5-0.00005*t)*((0.30-0.002*t)*math.sin((0.00050+0.0000005*t)*t) + (0.15+0.001*t)*math.sin((0.000004-0.000001*t)*t))),
            local_dt=data_dt
        )
        x_test = x_test.unsqueeze(0) / x_abs_max
        u_test = u_test.unsqueeze(0) / u_abs_max

        with torch.no_grad():
            # ensure model on cpu for inference determinism
            device = torch.device('cpu')
            best_model.to(device)
            # set the model internal dt to the requested test_dt (mismatch with data_dt)
            best_model.set_model_dt(test_dt)

            x_pred = best_model(u_test.to(device))[0].squeeze().numpy()
            test_true = x_test[0].squeeze().numpy()
            test_mse = float(((x_pred - test_true)**2).mean())
            spearman_test = spearmanr(test_true, x_pred)

        results[test_dt] = {
            'test_pred': x_pred,
            'test_true': test_true,
            'test_mse': test_mse,
            'spearman_test': spearman_test
        }
        print(f" -> test_dt={test_dt} | mse={test_mse:.3e} | spearman={spearman_test:.3f}")

    # Save results and metadata
    npz_name = os.path.join(results_dir, f"Sys_1_PhyGRU_modeldt{train_dt}_results.npz")
    save_dict = {
        'train_model_dt': np.array([train_dt]),
        'test_model_dts': np.array(test_dts),
        'data_dt': np.array([data_dt])
    }
    for td in results:
        save_dict[f'modeldt_{td}_pred'] = results[td]['test_pred']
        save_dict[f'modeldt_{td}_true'] = results[td]['test_true']
        save_dict[f'modeldt_{td}_mse'] = np.array([results[td]['test_mse']])
        save_dict[f'modeldt_{td}_spearman'] = np.array([results[td]['spearman_test']])

    save_dict['model'] = np.array([key])
    save_dict['ckpt'] = np.array([ckpt_path])
    save_dict['best_val_loss'] = np.array([info['best_val_loss']])
    save_dict['best_epoch'] = np.array([info['best_epoch']])

    np.savez(npz_name, **save_dict)
    print(f"Saved experiment results to {npz_name}")

    # Print compact resume table
    print("\nResume table for trained model and evaluations:")
    print(f"Model: {key} | ckpt: {ckpt_path} | best_epoch: {info['best_epoch']} | best_val_loss: {info['best_val_loss']:.3e}")
    for td in results:
        print(f" - model_dt={td} | test_mse={results[td]['test_mse']:.3e} | spearman={results[td]['spearman_test']:.3f}")

    return npz_name, ckpt_path, results

# ---------------------------
# Main: compute DT_critical and run experiments
# ---------------------------
if __name__ == '__main__':
    # analytic critical dt using initial parameters (a=0.5,b=0.6,c=0.7)
    a0, b0, c0 = 0.5, 0.6, 0.7
    dt_crit = compute_dt_critical(a0, b0, c0)
    print(f"Analytical explicit-Euler DT_critical (using a={a0}, b={b0}, c={c0}) = {dt_crit:.6f} s")

    # experiment model_dt values
    model_dt_good  = 0.01  # matches data sampling
    model_dt_small = 0.005 # model using smaller dt than data (interesting corner case)
    model_dt_bad   = 0.90   # much larger than data_dt

    # Run experiment: train with model_dt_bad, test on [model_dt_good, model_dt_bad, model_dt_small]
    npz_bad, ckpt_bad, res_bad = run_phygru_dt_experiments(model_dt_bad, [model_dt_good, model_dt_bad, model_dt_small])

    # Run experiment: train with model_dt_good (matches data), test on same set
    npz_good, ckpt_good, res_good = run_phygru_dt_experiments(model_dt_good, [model_dt_good, model_dt_bad, model_dt_small])

    # Save merged summary
    merged = {
        'dt_crit': np.array([dt_crit]),
        'model_dt_good': np.array([model_dt_good]),
        'model_dt_small': np.array([model_dt_small]),
        'model_dt_bad': np.array([model_dt_bad]),
        'npz_good': np.array([npz_good]),
        'npz_bad': np.array([npz_bad]),
        'ckpt_good': np.array([ckpt_good]),
        'ckpt_bad': np.array([ckpt_bad])
    }
    merged_npz = os.path.join('results_npz', 'Sys_1_PhyGRU_modeldt_experiment_summary.npz')
    np.savez(merged_npz, **merged)
    print(f"Merged summary saved to {merged_npz}")
    print('All done. Check checkpoints/ and results_npz/ for artifacts.')
