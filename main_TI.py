# [full script as requested with only the directed modifications]

import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


# System hardware info & Env
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
random.seed(0)
np.random.seed(0)

# =====================================
# Utilities
# =====================================
dt = 0.010
T  = 6000

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inference_time(model, u):
    with torch.no_grad():
        t0 = time.time()
        _ = model(u)
        t1 = time.time()
    B, TT, _ = u.shape
    return (t1 - t0) / (B * TT)

# Spearman correlation 
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
        # average rank for ties: 1-based ranks
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

# =====================================
# Dataset generators
# =====================================
def generate_data_1(u_fn):
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

def generate_data_2(u_fn):
    x, xd = 0.0, 0.0
    xs, us = [], []
    for t in range(T):
        u = u_fn(t)
        xdd = (u - 0.5 * xd - 0.2 * x + math.tanh(x*xd))
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
        xddd = u - 3 * xdd - 2*math.tanh(xd) - 0.1 * math.tanh(x) 
        xdd += dt * xddd
        xd  += dt * xdd
        x   += dt * xd
        xs.append([x])
        us.append([u])
    return torch.tensor(xs), torch.tensor(us)

# =====================================
# Standard GRU
# =====================================
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bias=True)
        self.fc  = nn.Linear(hidden_dim, 1)

    def forward(self, u):
        h, _ = self.gru(u)
        return self.fc(h)

# =====================================
# PhyGRU
# =====================================
class PhyGRUCell(nn.Module):
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
        phys_dot = self.physics_law(state[:, :self.state_dim], u)
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
        state = torch.zeros(B, self.state_dim + self.latent_dim, dtype=u_seq.dtype, device=u_seq.device)
        ys = []
        for t in range(Tt):
            state = self.cell(state, u_seq[:, t])
            ys.append(state[:, 0:1])
        return torch.stack(ys, dim=1)

# =====================================
# Physics Law (Prior)
# =====================================
class MassSpringDamperLaw(nn.Module):
    def __init__(self, learn_a=True, learn_b=True, learn_c=True):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5), requires_grad=learn_a)
        self.b = nn.Parameter(torch.tensor(0.6), requires_grad=learn_b)
        self.c = nn.Parameter(torch.tensor(0.7), requires_grad=learn_c)

    def forward(self, state, u):
        x, xd = state[:, 0], state[:, 1]
        u_s = u.squeeze()
        xdd = (u_s - self.b * xd - self.c * x) / (self.a + 1e-12)
        return torch.stack([xd, xdd], dim=1)

# =====================================
# Training with validation and checkpointing
# =====================================
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

        # Default teacher-forcing training loss (works for both F==1 and F==4,
        # assuming for F==4 the provided features already contain tanh'd past states)
        y_train_pred = model(u_train)
        teacher_loss = loss_fn(y_train_pred, x_train)

        # If this is an augmented-input GRU-like model (features==4 and has GRU+FC),
        # also compute a self-fed (closed-loop) training loss and combine them.
        B, Tt, F = u_train.shape
        if F == 4 and hasattr(model, "gru") and hasattr(model, "fc"):
            # self-fed closed-loop training rollout
            preds = torch.zeros(B, Tt, 1, dtype=x_train.dtype, device=device)
            for b in range(B):
                hidden = None
                # initialize from ground truth at t=0
                x_prev = float(x_train[b, 0, 0].item())
                xd_prev = 0.0
                xdd_prev = 0.0
                for t in range(Tt):
                    u_t = float(u_train[b, t, 0].item())  # u is first channel in the provided features
                    # apply tanh to past states before feeding to network (to limit derivatives)
                    x_in = float(np.tanh(x_prev))
                    xd_in = float(np.tanh(xd_prev))
                    xdd_in = float(np.tanh(xdd_prev))
                    inp = torch.tensor([[[u_t, x_in, xd_in, xdd_in]]], dtype=u_train.dtype, device=device)  # (1,1,4)
                    out, hidden = model.gru(inp, hidden)
                    y_t = model.fc(out)
                    x_pred_t = float(y_t[0, 0, 0].item())
                    preds[b, t, 0] = x_pred_t
                    # update derivatives from raw predictions (finite differences)
                    xd = (x_pred_t - x_prev) / dt
                    xdd = (xd - xd_prev) / dt
                    x_prev = x_pred_t
                    xd_prev = xd
                    xdd_prev = xdd
            selffed_loss = loss_fn(preds, x_train)
            # combine teacher and self-fed losses (average to keep scale similar)
            train_loss = 0.5 * (teacher_loss + selffed_loss)
        else:
            train_loss = teacher_loss

        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            # -----------------------------
            # VALIDATION IN SELF-FED MODE
            # For models that accept augmented inputs (features per time >1, e.g. 4),
            # perform closed-loop (self-fed) rollouts where only u(t) comes from data
            # and past states (x, xd, xdd) are taken from previous predictions.
            # For models that accept only u(t) (features == 1), the standard evaluation
            # model(u_val) already matches closed-loop behavior and is used as-is.
            # -----------------------------
            Bv, Ttv, Fv = u_val.shape
            if Fv == 4 and hasattr(model, "gru") and hasattr(model, "fc"):
                # self-fed closed-loop validation for augmented-input GRU-like models
                preds = torch.zeros(Bv, Ttv, 1, dtype=x_val.dtype, device=device)
                # iterate over batch
                for b in range(Bv):
                    hidden = None
                    # initialize from ground truth at t=0 
                    x_prev = float(x_val[b, 0, 0].item())
                    xd_prev = 0.0
                    xdd_prev = 0.0
                    for t in range(Ttv):
                        u_t = float(u_val[b, t, 0].item())  # first channel is u(t)
                        # apply tanh to past states before feeding to network (to limit derivatives)
                        x_in = float(np.tanh(x_prev))
                        xd_in = float(np.tanh(xd_prev))
                        xdd_in = float(np.tanh(xdd_prev))
                        inp = torch.tensor([[[u_t, x_in, xd_in, xdd_in]]], dtype=u_val.dtype, device=device)  # (1,1,4)
                        out, hidden = model.gru(inp, hidden)  # (1,1,hidden)
                        y_t = model.fc(out)  # (1,1,1)
                        x_pred_t = float(y_t[0, 0, 0].item())
                        preds[b, t, 0] = x_pred_t
                        # update derivatives
                        xd = (x_pred_t - x_prev) / dt
                        xdd = (xd - xd_prev) / dt
                        x_prev = x_pred_t
                        xd_prev = xd
                        xdd_prev = xdd
                y_val_pred = preds
                val_loss = loss_fn(y_val_pred, x_val).item()
            else:
                # default evaluation (works for standard GRU with F==1 and for PhyGRU)
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

# =====================================
# Main Benchmark Loop
# =====================================
if __name__ == "__main__":

    sns.set_palette("colorblind")
    datasets = [
        ("Sys_1", generate_data_1),
        ("Sys_2", generate_data_2),
        ("Sys_3", generate_data_3),
    ]

    latent_dims  = [0, 1, 2, 3]
    hidden_sizes = [1, 2, 4, 8, 32]

    results_dir     = "results_npz"
    checkpoints_dir = "checkpoints"
    os.makedirs(results_dir,     exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    EPOCHS = 150
    LR     = 5e-3

    for name, data_fn in datasets:

        print(f"\n==============================\n        {name}\n==============================")

        x_train, u_train = data_fn(
        lambda t: math.tanh((0.3-0.00005*t)*((0.25-0.001*t)*math.sin((0.00007+0.0000001*t)*t) + (0.10+0.001*t)*math.sin((0.000001-0.000001*t)*t)))
        )
        x_train = x_train.unsqueeze(0)
        u_train = u_train.unsqueeze(0)

        x_val, u_val = data_fn(
        lambda t: math.tanh((0.4-0.00005*t)*((0.35-0.003*t)*math.sin((0.00009+0.0000002*t)*t) + (0.15+0.001*t)*math.sin((0.000003-0.000001*t)*t)))
        )
        x_val = x_val.unsqueeze(0)
        u_val = u_val.unsqueeze(0)

        x_test, u_test = data_fn(
        lambda t: math.tanh((0.5-0.00005*t)*((0.30-0.002*t)*math.sin((0.00050+0.0000005*t)*t) + (0.15+0.001*t)*math.sin((0.000004-0.000001*t)*t)))
        )
        x_test = x_test.unsqueeze(0)
        u_test = u_test.unsqueeze(0)

        # -------------------------
        # NORMALIZATION: scale x and u by their absolute max across train only (apply same to val/test)
        # -------------------------
        # compute absolute max for x (from training set only)
        x_abs_max = float(torch.max(torch.abs(x_train)).item())
        if x_abs_max == 0.0:
            x_abs_max = 1.0

        # compute absolute max for u (from training set only)
        u_abs_max = float(torch.max(torch.abs(u_train)).item())
        if u_abs_max == 0.0:
            u_abs_max = 1.0

        # apply normalization (train scalers applied to val/test)
        x_train = x_train / x_abs_max
        x_val   = x_val   / x_abs_max
        x_test  = x_test  / x_abs_max

        u_train = u_train / u_abs_max
        u_val   = u_val   / u_abs_max
        u_test  = u_test  / u_abs_max
        # -------------------------

        # -------------------------
        # Build augmented inputs for the GRU_obs (u, x(t-1), xd(t-1), xdd(t-1))
        # Apply tanh to past state features to limit derivative magnitudes
        # -------------------------
        def build_augmented_inputs(x_tensor, u_tensor):
            # x_tensor, u_tensor: shape (1, T, 1)
            x_np = x_tensor.squeeze(0).squeeze(-1).numpy()  # (T,)
            u_np = u_tensor.squeeze(0).squeeze(-1).numpy()  # (T,)

            # compute first derivative xd and second derivative xdd 
            xd  = np.zeros_like(x_np)
            xdd = np.zeros_like(x_np)
            # forward differences for derivatives (t>0)
            for t in range(1, x_np.size):
                xd[t] = (x_np[t] - x_np[t-1]) / dt
            for t in range(1, x_np.size):
                xdd[t] = (xd[t] - xd[t-1]) / dt

            # build augmented features: at time t, include u[t], tanh(x[t-1]), tanh(xd[t-1]), tanh(xdd[t-1])
            feats = np.zeros((x_np.size, 4), dtype=np.float32)
            for t in range(x_np.size):
                if t == 0:
                    x_prev   = 0.0
                    xd_prev  = 0.0
                    xdd_prev = 0.0
                else:
                    x_prev   = x_np[t-1]
                    xd_prev  = xd[t-1]
                    xdd_prev = xdd[t-1]
                feats[t, 0] = u_np[t]
                # apply tanh to past states (as requested)
                feats[t, 1] = np.tanh(x_prev)
                feats[t, 2] = np.tanh(xd_prev)
                feats[t, 3] = np.tanh(xdd_prev)
            return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # shape (1, T, 4)

        u_train_aug = build_augmented_inputs(x_train, u_train)
        u_val_aug   = build_augmented_inputs(x_val,   u_val)
        u_test_aug  = build_augmented_inputs(x_test,  u_test)

        all_preds = {"GRTH": x_test.squeeze().numpy(), "U": u_test.squeeze().numpy(), "GRTH_VAL": x_val.squeeze().numpy()}
        model_summary = []

        # ---------- GRU ----------
        for hs in hidden_sizes:
            key = f"GRU_h{hs}"
            print(f"\nTraining {key}  (params = {count_parameters(GRUModel(1, hs))})")

            gru = GRUModel(1, hs)
            ckpt_path = os.path.join(checkpoints_dir, f"{name}_{key}_best.pt")

            info = train_with_validation(
                gru, u_train, x_train, u_val, x_val,
                save_path=ckpt_path, epochs=EPOCHS, lr=LR, verbose=False
            )

            best_model = GRUModel(1, hs)
            best_model.load_state_dict(torch.load(ckpt_path))
            best_model.eval()
            with torch.no_grad():
                x_pred = best_model(u_test)[0]
                test_mse = ((x_pred - x_test[0])**2).mean().item()

                # Validation (standard GRU: model(u_val) is closed-loop style natively)
                val_pred_arr = best_model(u_val)[0].squeeze().numpy()
                val_true_arr = x_val[0].squeeze().numpy()

                # Spearman correlations: validation and test vs ground truth (GRTH)
                val_pred  = val_pred_arr
                val_true  = val_true_arr
                test_pred = x_pred.squeeze().numpy()
                test_true = x_test[0].squeeze().numpy()

                spearman_val  = spearmanr(val_true, val_pred)
                spearman_test = spearmanr(test_true, test_pred)

            all_preds[key] = test_pred
            all_preds[f"{key}_val"] = val_pred_arr

            model_summary.append({
                "model": key,
                "best_val_loss": info["best_val_loss"],
                "best_epoch":    info["best_epoch"],
                "test_mse":      test_mse,
                "spearman_val":  spearman_val,
                "spearman_test": spearman_test,
                "ckpt":          ckpt_path
            })

            # Measure inference time for Sys_1 only
            if name == "Sys_1":
                inf_time_ms = inference_time(best_model, u_test) * 1000
                print(f"{key} | Best val epoch: {info['best_epoch']} | Best val loss: {info['best_val_loss']:.3e} | Test MSE: {test_mse:.3e} | Spearman(val/test) = {spearman_val:.3f}/{spearman_test:.3f} | Inference/sample = {inf_time_ms:.3f} ms")
            else:
                print(f"{key} | Best val epoch: {info['best_epoch']} | Best val loss: {info['best_val_loss']:.3e} | Test MSE: {test_mse:.3e} | Spearman(val/test) = {spearman_val:.3f}/{spearman_test:.3f}")

        # ---------- GRU with observed states (u, x(t-1), xd(t-1), xdd(t-1)) ----------
        for hs in hidden_sizes:
            key = f"GRU_obs_h{hs}"
            print(f"\nTraining {key}  (params = {count_parameters(GRUModel(4, hs))})")

            gru_obs = GRUModel(4, hs)
            ckpt_path = os.path.join(checkpoints_dir, f"{name}_{key}_best.pt")

            info = train_with_validation(
                gru_obs, u_train_aug, x_train, u_val_aug, x_val,
                save_path=ckpt_path, epochs=EPOCHS, lr=LR, verbose=False
            )

            best_model = GRUModel(4, hs)
            best_model.load_state_dict(torch.load(ckpt_path))
            best_model.eval()
            with torch.no_grad():
                # ----------------------------
                # TEST: pure self-fed (closed-loop) evaluation
                # ----------------------------
                device = next(best_model.parameters()).device
                Tt = x_test.shape[1]
                preds = np.zeros(Tt, dtype=np.float32)

                # initialize from ground truth at t=0 
                x_prev = float(x_test[0, 0, 0].item())
                xd_prev  = 0.0
                xdd_prev = 0.0

                hidden = None  # so GRU hidden state is preserved across time-steps

                for t in range(Tt):
                    u_t = float(u_test[0, t, 0].item())
                    # apply tanh to past states before feeding
                    x_in = float(np.tanh(x_prev))
                    xd_in = float(np.tanh(xd_prev))
                    xdd_in = float(np.tanh(xdd_prev))
                    inp = torch.tensor([[[u_t, x_in, xd_in, xdd_in]]], dtype=torch.float32, device=device)  # shape (1,1,4)
                    out, hidden = best_model.gru(inp, hidden)  # out: (1,1,hidden_dim)
                    y_t = best_model.fc(out)  # (1,1,1)
                    x_pred_t = float(y_t[0, 0, 0].item())
                    preds[t] = x_pred_t

                    # update derivatives based on predictions (finite differences)
                    xd  = (x_pred_t - x_prev) / dt
                    xdd = (xd - xd_prev) / dt

                    # shift for next step (self-feeding)
                    x_prev   = x_pred_t
                    xd_prev  = xd
                    xdd_prev = xdd

                # Convert preds to array for metrics
                test_pred_arr = preds  # (T,)
                test_true_arr = x_test[0].squeeze().numpy()  # (T,)

                test_mse = float(((test_pred_arr - test_true_arr)**2).mean())

                # ----------------------------
                # VALIDATION: closed-loop validation prediction (self-fed)
                # ----------------------------
                Tval = x_val.shape[1]
                val_preds = np.zeros(Tval, dtype=np.float32)
                x_prev_v  = float(x_val[0, 0, 0].item())
                xd_prev_v  = 0.0
                xdd_prev_v = 0.0
                hidden_v = None
                for t in range(Tval):
                    u_tv  = float(u_val[0, t, 0].item())
                    # apply tanh to past states before feeding
                    x_in_v = float(np.tanh(x_prev_v))
                    xd_in_v = float(np.tanh(xd_prev_v))
                    xdd_in_v = float(np.tanh(xdd_prev_v))
                    inp_v = torch.tensor([[[u_tv, x_in_v, xd_in_v, xdd_in_v]]], dtype=torch.float32, device=device)
                    out_v, hidden_v = best_model.gru(inp_v, hidden_v)
                    y_tv  = best_model.fc(out_v)
                    x_pred_tv = float(y_tv[0, 0, 0].item())
                    val_preds[t] = x_pred_tv
                    xd_v  = (x_pred_tv - x_prev_v) / dt
                    xdd_v = (xd_v - xd_prev_v) / dt
                    x_prev_v   = x_pred_tv
                    xd_prev_v  = xd_v
                    xdd_prev_v = xdd_v

                val_pred_arr = val_preds
                val_true_arr = x_val[0].squeeze().numpy()

                spearman_val  = spearmanr(val_true_arr,  val_pred_arr)
                spearman_test = spearmanr(test_true_arr, test_pred_arr)

        
                all_preds[key] = test_pred_arr
                all_preds[f"{key}_val"] = val_pred_arr
                x_pred = torch.tensor(test_pred_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            model_summary.append({
                "model": key,
                "best_val_loss": info["best_val_loss"],
                "best_epoch":    info["best_epoch"],
                "test_mse":      test_mse,
                "spearman_val":  spearman_val,
                "spearman_test": spearman_test,
                "ckpt":          ckpt_path
            })

            # Measure inference time for Sys_1 only 
            if name == "Sys_1":
                inf_time_ms = inference_time(best_model, u_test_aug) * 1000
                print(f"{key} | Best val epoch: {info['best_epoch']} | Best val loss: {info['best_val_loss']:.3e} | Test MSE: {test_mse:.3e} | Spearman(val/test) = {spearman_val:.3f}/{spearman_test:.3f} | Inference/sample = {inf_time_ms:.3f} ms")
            else:
                print(f"{key} | Best val epoch: {info['best_epoch']} | Best val loss: {info['best_val_loss']:.3e} | Test MSE: {test_mse:.3e} | Spearman(val/test) = {spearman_val:.3f}/{spearman_test:.3f}")

        # ---------- PhyGRU ----------
        for ld in latent_dims:
            key = f"PhyGRU_l{ld}"
            print(f"\nTraining {key}")

            phygru = PhyGRU(MassSpringDamperLaw(), 2, 1, latent_dim=ld)
            ckpt_path = os.path.join(checkpoints_dir, f"{name}_{key}_best.pt")

            info = train_with_validation(
                phygru, u_train, x_train, u_val, x_val,
                save_path=ckpt_path, epochs=EPOCHS, lr=LR, verbose=False
            )

            best_model = PhyGRU(MassSpringDamperLaw(), 2, 1, latent_dim=ld)
            best_model.load_state_dict(torch.load(ckpt_path))
            best_model.eval()
            with torch.no_grad():
                x_pred = best_model(u_test)[0]
                test_mse = ((x_pred - x_test[0])**2).mean().item()

                # For PhyGRU, model(u_val) is already a native closed-loop rollout 
                val_pred_arr = best_model(u_val)[0].squeeze().numpy()
                val_true_arr = x_val[0].squeeze().numpy()
                test_pred = x_pred.squeeze().numpy()
                test_true = x_test[0].squeeze().numpy()

                spearman_val = spearmanr(val_true_arr, val_pred_arr)
                spearman_test = spearmanr(test_true, test_pred)

            all_preds[key] = test_pred
            all_preds[f"{key}_val"] = val_pred_arr

            model_summary.append({
                "model": key,
                "best_val_loss": info["best_val_loss"],
                "best_epoch":    info["best_epoch"],
                "test_mse":      test_mse,
                "spearman_val":  spearman_val,
                "spearman_test": spearman_test,
                "ckpt":          ckpt_path
            })

            # Measure inference time for Sys_1 only
            if name == "Sys_1":
                inf_time_ms = inference_time(best_model, u_test) * 1000
                print(f"{key} | Best val epoch: {info['best_epoch']} | Best val loss: {info['best_val_loss']:.3e} | Test MSE: {test_mse:.3e} | Spearman(val/test) = {spearman_val:.3f}/{spearman_test:.3f} | Inference/sample = {inf_time_ms:.3f} ms")
            else:
                print(f"{key} | Best val epoch: {info['best_epoch']} | Best val loss: {info['best_val_loss']:.3e} | Test MSE: {test_mse:.3e} | Spearman(val/test) = {spearman_val:.3f}/{spearman_test:.3f}")

        npz_file = os.path.join(results_dir, f"{name}_predictions.npz")
        model_info = {
            "models":        np.array([m["model"]         for m in model_summary]),
            "best_val_loss": np.array([m["best_val_loss"] for m in model_summary]),
            "best_epoch":    np.array([m["best_epoch"]    for m in model_summary]),
            "test_mse":      np.array([m["test_mse"]      for m in model_summary]),
            "spearman_val":  np.array([m["spearman_val"]  for m in model_summary]),
            "spearman_test": np.array([m["spearman_test"] for m in model_summary]),
            "ckpt":          np.array([m["ckpt"]          for m in model_summary])
        }
        np.savez(npz_file, **all_preds, **model_info)
        print(f"Saved predictions and model summary to {npz_file}")

        print("\nSummary of best checkpoints for dataset:", name)
        for m in model_summary:
            print(f" - {m['model']}: best epoch = {m['best_epoch']} | best val loss = {m['best_val_loss']:.3e} | test mse = {m['test_mse']:.3e} | Spearman(val/test) = {m['spearman_val']:.3f}/{m['spearman_test']:.3f} | ckpt: {m['ckpt']}")

    print("\nAll done. Best model checkpoints are stored in the `checkpoints/` folder (only best-by-validation are kept).")
