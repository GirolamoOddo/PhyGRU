import torch
import torch.nn as nn
import numpy as np
import math
import os
from scipy.stats import spearmanr

# ============================================================
# PARAMS
# ============================================================
dt = 0.01
T  = 6000
hidden = [8,16,32]
latent = [0,1,2]
datasets = [
    ("Sys1", "generate_data_1"),
    ("Sys2", "generate_data_2"),
    ("Sys3", "generate_data_3"),
]

os.makedirs("results_npz", exist_ok=True)

# ============================================================
# DATA GENERATION FUNCTIONS
# ============================================================
def generate_data_1(u_fn):
    x, xd = 0.0, 0.0
    xs, us = [], []
    for t in range(T):
        u = u_fn(t)
        xdd = u - (0.5+0.00009*t)*xd - (0.2+1e-7*t**2)*x
        xd += dt*xdd
        x  += dt*xd
        xs.append([x]); us.append([u])
    return torch.tensor(xs), torch.tensor(us)

def generate_data_2(u_fn):
    x, xd = 0.0, 0.0
    xs, us = [], []
    for t in range(T):
        u = u_fn(t)
        xdd = u - (0.5+0.00009*t)*xd - (0.2+1e-7*t**2)*x + (1+0.00009*t)*math.tanh(x*xd)
        xd += dt*xdd
        x  += dt*xd
        xs.append([x]); us.append([u])
    return torch.tensor(xs), torch.tensor(us)

def generate_data_3(u_fn):
    x, xd, xdd = 0.0, 0.0, 0.0
    xs, us = [], []
    for t in range(T):
        u = u_fn(t)
        xddd = u - (3-0.0002*t)*xdd - (2+0.0009*t)*math.tanh(xd) - (0.1+3e-7*t**2)*math.tanh(x)
        xdd += dt*xddd
        xd  += dt*xdd
        x   += dt*xd
        xs.append([x]); us.append([u])
    return torch.tensor(xs), torch.tensor(us)

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class GRU_GRU(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.g1 = nn.GRU(1,h,batch_first=True)
        self.f1 = nn.Linear(h,1)
        self.g2 = nn.GRU(1,h,batch_first=True)
        self.f2 = nn.Linear(h,1)
    def forward(self,u):
        y1,_ = self.g1(u)
        y1 = self.f1(y1)
        y2,_ = self.g2(y1)
        return self.f2(y2)

class MassSpringDamperLaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.6))
        self.c = nn.Parameter(torch.tensor(0.7))
    def forward(self, state, u):
        x, xd = state[:,0], state[:,1]
        u = u.squeeze()
        xdd = (u - self.b*xd - self.c*x)/(self.a+1e-12)
        return torch.stack([xd, xdd],1)

class PhyGRUCell(nn.Module):
    def __init__(self, physics, latent):
        super().__init__()
        self.physics = physics
        self.latent = latent
        dim = 2 + latent
        if latent>0:
            self.latent_dyn = nn.Linear(dim+1, latent)
        self.z = nn.Sequential(nn.Linear(dim+1, dim), nn.Sigmoid())
    def forward(self, s, u):
        phys = self.physics(s[:,:2], u)
        s_phys = s[:,:2]+dt*phys
        if self.latent>0:
            ldot = self.latent_dyn(torch.cat([s,u],1))
            lnext = s[:,2:]+dt*ldot
            cand = torch.cat([s_phys, lnext],1)
        else:
            cand = s_phys
        z = self.z(torch.cat([s,u],1))
        return z*cand + (1-z)*s

class PhyGRU(nn.Module):
    def __init__(self, physics, latent):
        super().__init__()
        self.cell = PhyGRUCell(physics, latent)
        self.latent = latent
    def forward(self,u):
        B,Tt,_ = u.shape
        s = torch.zeros(B,2+self.latent)
        ys=[]
        for t in range(Tt):
            s = self.cell(s,u[:,t])
            ys.append(s[:,0:1])
        return torch.stack(ys,1)

class PhyGRU_GRU(nn.Module):
    def __init__(self, latent,h):
        super().__init__()
        self.phy = PhyGRU(MassSpringDamperLaw(), latent)
        self.g = nn.GRU(1,h,batch_first=True)
        self.f = nn.Linear(h,1)
    def forward(self,u):
        y1 = self.phy(u)
        y,_ = self.g(y1)
        return self.f(y)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mse(y_true,y_pred):
    return np.mean((y_true - y_pred)**2)

def evaluate_checkpoint(model, ckpt_path, u_tr,x_tr,u_v,x_v,u_te,x_te):
    model.load_state_dict(torch.load(ckpt_path,map_location="cpu"))
    model.eval()
    with torch.no_grad():
        y_tr = model(u_tr)[0].squeeze().numpy()
        y_v  = model(u_v )[0].squeeze().numpy()
        y_te = model(u_te)[0].squeeze().numpy()
    x_tr = x_tr.squeeze().numpy()
    x_v  = x_v.squeeze().numpy()
    x_te = x_te.squeeze().numpy()
    out = {
        "train_mse": mse(x_tr, y_tr),
        "val_mse":   mse(x_v,  y_v),
        "test_mse":  mse(x_te, y_te),
        "sp_val":    spearmanr(x_v,  y_v).correlation,
        "sp_test":   spearmanr(x_te, y_te).correlation
    }
    return out

def generate_all_sets(gen):
    x_tr,u_tr = gen(lambda t: math.tanh((0.3-0.00005*t)*((0.25-0.001*t)*math.sin((0.00007+0.0000001*t)*t) + (0.10+0.001*t)*math.sin((0.000001-0.000001*t)*t))))
    x_v,u_v   = gen(lambda t: math.tanh((0.4-0.00005*t)*((0.35-0.003*t)*math.sin((0.00009+0.0000002*t)*t) + (0.15+0.001*t)*math.sin((0.000003-0.000001*t)*t))))
    x_te,u_te = gen(lambda t: math.tanh((0.5-0.00005*t)*((0.30-0.002*t)*math.sin((0.00050+0.0000005*t)*t) + (0.15+0.001*t)*math.sin((0.000004-0.000001*t)*t))))
    x_tr=x_tr.unsqueeze(0); u_tr=u_tr.unsqueeze(0)
    x_v =x_v.unsqueeze(0);  u_v=u_v.unsqueeze(0)
    x_te=x_te.unsqueeze(0); u_te=u_te.unsqueeze(0)
    xm = torch.max(torch.abs(x_tr)); um = torch.max(torch.abs(u_tr))
    x_tr/=xm; x_v/=xm; x_te/=xm
    u_tr/=um; u_v/=um; u_te/=um
    return x_tr,u_tr,x_v,u_v,x_te,u_te


results_summary = {}

for name,gen_str in datasets:
    print(f"\n===== {name} =====")
    gen = globals()[gen_str]
    x_tr,u_tr,x_v,u_v,x_te,u_te = generate_all_sets(gen)
    results_summary[name] = {}

    # GRU
    for h in hidden:
        key = f"GRU_GRU_h{h}"
        ckpt = f"checkpoints/{name}_{key}.pt"
        if not os.path.exists(ckpt): continue
        model = GRU_GRU(h)
        stats = evaluate_checkpoint(model, ckpt, u_tr,x_tr,u_v,x_v,u_te,x_te)
        stats["params"] = count_trainable_params(model)
        stats["ckpt"] = ckpt
        results_summary[name][key] = stats

    # PhyGRU
    for l in latent:
        for h in hidden:
            key = f"PhyGRU_GRU_l{l}_h{h}"
            ckpt = f"checkpoints/{name}_{key}.pt"
            if not os.path.exists(ckpt): continue
            model = PhyGRU_GRU(l,h)
            stats = evaluate_checkpoint(model, ckpt, u_tr,x_tr,u_v,x_v,u_te,x_te)
            stats["params"] = count_trainable_params(model)
            stats["ckpt"] = ckpt
            results_summary[name][key] = stats

# save results
for name,data in results_summary.items():
    np.savez(f"results_npz/{name}_summary.npz", **{k:str(v) for k,v in data.items()})

print("\nALL DONE. Results saved in results_npz/")




