import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ================================
# CONFIG
# ================================
dt = 0.01
T = 5000
N_test = 36

sns.set_style("whitegrid")
sns.set_palette("colorblind")

# ================================
# 7DOF SIMULATOR 
# ================================
class Vehicle7DOFSimulator:
    def __init__(self):
        self.m = 1500.0
        self.Iz = 2500.0
        self.Iw = 1.8
        self.lf = 1.25
        self.lr = 1.55
        self.Cf = 70000.0
        self.Cr = 80000.0
        self.Cd = 0.35
        self.Crr = 40.0
        self.Fx_max = 4500.0
        self.Fb_max = 9000.0
        self.T_drive_max = 3500.0
        self.T_brake_max = 6500.0
        self.v_min = 1.0
        self.Rw = 0.31

    def step(self, x, u):
        vx, vy, r, w_fl, w_fr, w_rl, w_rr = x
        delta, throttle, brake = u

        vx_safe = np.sign(vx) * max(abs(vx), self.v_min)
        if abs(vx_safe) < self.v_min:
            vx_safe = self.v_min

        alpha_f = delta - math.atan2(vy + self.lf * r, vx_safe)
        alpha_r = -math.atan2(vy - self.lr * r, vx_safe)

        Fyf = 2.0 * self.Cf * np.tanh(alpha_f)
        Fyr = 2.0 * self.Cr * np.tanh(alpha_r)

        Fx = self.Fx_max * throttle - self.Fb_max * brake - self.Cd * vx * abs(vx) - self.Crr * np.tanh(vx)

        vx_dot = (Fx - Fyf * np.sin(delta) + self.m * vy * r) / self.m
        vy_dot = (Fyf * np.cos(delta) + Fyr - self.m * vx * r) / self.m
        r_dot = (self.lf * Fyf * np.cos(delta) - self.lr * Fyr) / self.Iz

        ax = vx_dot - vy * r
        ay = vy_dot + vx * r

        # wheel dynamics
        T_drive = self.T_drive_max * throttle
        T_brake = self.T_brake_max * brake

        Fx_rear = 0.6 * Fx
        Fx_front = 0.4 * Fx

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


# ================================
# BICYCLE MODEL
# ================================
class BicycleModel:
    def __init__(self):
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

    def step(self, state, u):
        vx, vy, r = state
        delta, throttle, brake = u

        vx_safe = vx if abs(vx) >= self.v_min else self.v_min

        alpha_f = delta - np.arctan2(vy + self.lf * r, vx_safe)
        alpha_r = -np.arctan2(vy - self.lr * r, vx_safe)

        Fyf = 2.0 * self.Cf * np.tanh(alpha_f)
        Fyr = 2.0 * self.Cr * np.tanh(alpha_r)

        Fx = self.Fx_max * throttle - self.Fb_max * brake - self.Cd * vx * abs(vx) - self.Crr * np.tanh(vx)

        vx_dot = (Fx - Fyf * np.sin(delta) + self.m * vy * r) / self.m
        vy_dot = (Fyf * np.cos(delta) + Fyr - self.m * vx * r) / self.m
        r_dot = (self.lf * Fyf * np.cos(delta) - self.lr * Fyr) / self.Iz

        ax = vx_dot - vy * r
        ay = vy_dot + vx * r

        s_dot = np.array([vx_dot, vy_dot, r_dot], dtype=np.float32)
        y = np.array([ax, ay, r], dtype=np.float32)

        return s_dot, y


# ================================
# CONTROL GENERATOR 
# ================================
def smooth_random_controls(T, rng):
    t = np.arange(T) * dt

    f1 = rng.uniform(0.05, 0.20)
    f2 = rng.uniform(0.15, 0.45)
    p1 = rng.uniform(0, 2*np.pi)
    p2 = rng.uniform(0, 2*np.pi)

    delta = 0.06*np.sin(2*np.pi*f1*t+p1) + 0.03*np.sin(2*np.pi*f2*t+p2)

    ft = rng.uniform(0.03, 0.15)
    pt = rng.uniform(0, 2*np.pi)
    throttle = 0.55 + 0.20*np.sin(2*np.pi*ft*t+pt)
    throttle += 0.05*np.sin(2*np.pi*(2*ft)*t+0.3)
    throttle = np.clip(throttle,0,1)

    fb = rng.uniform(0.02, 0.10)
    pb = rng.uniform(0, 2*np.pi)
    brake = 0.10*np.maximum(0,np.sin(2*np.pi*fb*t+pb))
    brake += 0.03*np.maximum(0,np.sin(2*np.pi*(2.5*fb)*t+1))
    brake = np.clip(brake,0,1)

    return np.stack([delta, throttle, brake], axis=1).astype(np.float32)


# ================================
# SIMULATION FUNCTIONS
# ================================
def simulate_7dof(u_seq, seed):
    rng = np.random.default_rng(seed)
    sim = Vehicle7DOFSimulator()

    x = np.array([15,0,0,50,50,50,50], dtype=np.float32)
    Y = []

    for t in range(u_seq.shape[0]):
        x, y = sim.step(x, u_seq[t])
        Y.append(y)

    return np.array(Y)


def simulate_bicycle(u_seq):
    model = BicycleModel()
    state = np.array([15.0,0.0,0.0], dtype=np.float32)
    Y = []

    for t in range(u_seq.shape[0]):
        s_dot, y = model.step(state, u_seq[t])
        state += dt * s_dot
        Y.append(y)

    return np.array(Y)


# ================================
# POSE RECONSTRUCTION
# ================================
def reconstruct_pose(y):
    x,y_pos,psi = [0],[0],[0]
    vx,vy = 15.0,0.0

    for k in range(y.shape[0]):
        ax,ay,r = y[k]

        vx += dt*ax
        vy += dt*ay
        psi_new = psi[-1] + dt*r

        x.append(x[-1] + dt*(vx*np.cos(psi_new)-vy*np.sin(psi_new)))
        y_pos.append(y_pos[-1] + dt*(vx*np.sin(psi_new)+vy*np.cos(psi_new)))
        psi.append(psi_new)

    return np.stack([x[1:],y_pos[1:],psi[1:]],axis=1)


# ================================
# DATASET
# ================================
rng = np.random.default_rng(2001)

U_test = np.zeros((N_test,T,3),dtype=np.float32)
Y_gt = np.zeros((N_test,T,3),dtype=np.float32)
Y_bike = np.zeros_like(Y_gt)

for i in range(N_test):
    u = smooth_random_controls(T, rng)
    U_test[i] = u
    Y_gt[i] = simulate_7dof(u, seed=2001+i)
    Y_bike[i] = simulate_bicycle(u)


# ================================
# GRID PLOT 
# ================================
def plot_grid(y_gt_all, y_bike_all):
    fig, axes = plt.subplots(6,6,figsize=(14,14))

    for i, ax in enumerate(axes.flat):
        if i >= y_gt_all.shape[0]:
            ax.axis("off")
            continue

        pose_gt = reconstruct_pose(y_gt_all[i])
        pose_b  = reconstruct_pose(y_bike_all[i])

        ax.plot(pose_gt[:,0], pose_gt[:,1], lw=1.2, label="7DOF")
        ax.plot(pose_b[:,0],  pose_b[:,1],  lw=1.0, label="Bicycle")

        x_all = np.concatenate([pose_gt[:,0], pose_b[:,0]])
        y_all = np.concatenate([pose_gt[:,1], pose_b[:,1]])

        x_mid = 0.5*(x_all.min()+x_all.max())
        y_mid = 0.5*(y_all.min()+y_all.max())
        max_range = max(x_all.max()-x_all.min(), y_all.max()-y_all.min())/2

        ax.set_xlim(x_mid-max_range, x_mid+max_range)
        ax.set_ylim(y_mid-max_range, y_mid+max_range)

        ax.set_aspect("equal")
        
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")


    plt.tight_layout()
    plt.show()


# ================================
# RUN
# ================================
plot_grid(Y_gt, Y_bike)



# ================================
# MSE COMPUTATION
# ================================
def compute_mse(y_gt, y_pred):
    diff2 = (y_gt - y_pred) ** 2
    mse_total = diff2.mean()
    mse_per_channel = diff2.mean(axis=(0, 1))
    return mse_total, mse_per_channel


mse_total, mse_per_channel = compute_mse(Y_gt, Y_bike)

print("\n==============================")
print("BICYCLE vs 7DOF")
print("==============================")
print(f"Total MSE: {mse_total:.6e}")
print(f"Per-channel MSE [ax, ay, yaw_rate]: {mse_per_channel}")

