import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

# ─── Seed for reproducibility ───────────────────────────────────────────────
np.random.seed(42)

# ─── Simulation Parameters ──────────────────────────────────────────────────
DT          = 0.1       # time step (s)
SIM_TIME    = 20.0      # total simulation time (s)
V           = 1.0       # linear velocity (m/s)
OMEGA       = 0.314     # angular velocity (rad/s)  → ~circle of radius ~3.18m

# ─── Noise Parameters ───────────────────────────────────────────────────────
# Motion noise (process noise)
Q = np.diag([0.1, 0.1, np.deg2rad(2.0)]) ** 2   # [x, y, theta]

# Observation noise (range, bearing)
R = np.diag([0.2, np.deg2rad(5.0)]) ** 2          # [range, bearing]

# ─── Landmarks (x, y) ───────────────────────────────────────────────────────
LANDMARKS = np.array([
    [4.0,  0.0],
    [3.0,  3.0],
    [0.0,  4.0],
    [-3.0, 3.0],
    [-4.0, 0.0],
    [-3.0,-3.0],
    [0.0, -4.0],
    [3.0, -3.0],
])

# ─── Motion Model (Unicycle / Differential Drive) ───────────────────────────
def motion_model(state, v, omega, dt):
    x, y, theta = state
    x     += v * np.cos(theta) * dt
    y     += v * np.sin(theta) * dt
    theta += omega * dt
    return np.array([x, y, theta])

def jacobian_F(state, v, dt):
    """Jacobian of motion model w.r.t. state"""
    _, _, theta = state
    F = np.eye(3)
    F[0, 2] = -v * np.sin(theta) * dt
    F[1, 2] =  v * np.cos(theta) * dt
    return F

# ─── Observation Model (Range & Bearing) ────────────────────────────────────
def observation_model(state, landmark):
    dx = landmark[0] - state[0]
    dy = landmark[1] - state[1]
    r  = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx) - state[2]
    phi = np.arctan2(np.sin(phi), np.cos(phi))  # normalize to [-pi, pi]
    return np.array([r, phi])

def jacobian_H(state, landmark):
    """Jacobian of observation model w.r.t. state"""
    dx = landmark[0] - state[0]
    dy = landmark[1] - state[1]
    r2 = dx**2 + dy**2
    r  = np.sqrt(r2)
    H = np.array([
        [-dx/r,   -dy/r,   0],
        [ dy/r2,  -dx/r2, -1],
    ])
    return H

# ─── EKF Steps ───────────────────────────────────────────────────────────────
def ekf_predict(mu, Sigma, v, omega, dt):
    F     = jacobian_F(mu, v, dt)
    mu    = motion_model(mu, v, omega, dt)
    Sigma = F @ Sigma @ F.T + Q
    return mu, Sigma

def ekf_update(mu, Sigma, z, landmark):
    z_hat = observation_model(mu, landmark)
    H     = jacobian_H(mu, landmark)
    S     = H @ Sigma @ H.T + R
    K     = Sigma @ H.T @ np.linalg.inv(S)

    innov    = z - z_hat
    innov[1] = np.arctan2(np.sin(innov[1]), np.cos(innov[1]))  # normalize

    mu    = mu + K @ innov
    Sigma = (np.eye(3) - K @ H) @ Sigma
    return mu, Sigma

# ─── Helper: Draw Uncertainty Ellipse ────────────────────────────────────────
def get_ellipse(mu, Sigma, n_std=2.0):
    vals, vecs = np.linalg.eigh(Sigma[:2, :2])
    order  = vals.argsort()[::-1]
    vals   = np.abs(vals[order])
    vecs   = vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    return Ellipse(xy=mu[:2], width=width, height=height,
                   angle=angle, edgecolor='purple', fc='None',
                   lw=1.5, linestyle='--', alpha=0.6)

# ─── Run Simulation ───────────────────────────────────────────────────────────
steps     = int(SIM_TIME / DT)
true_state  = np.array([0.0, 0.0, 0.0])
pred_state  = np.array([0.0, 0.0, 0.0])
ekf_state   = np.array([0.0, 0.0, 0.0])
Sigma       = np.diag([0.1, 0.1, np.deg2rad(5.0)]) ** 2

true_traj  = [true_state.copy()]
pred_traj  = [pred_state.copy()]
ekf_traj   = [ekf_state.copy()]
ellipses   = []
sigmas     = [Sigma.copy()]

for _ in range(steps):
    # ── True state (with motion noise) ──
    noise      = np.random.multivariate_normal([0, 0, 0], Q)
    true_state = motion_model(true_state, V, OMEGA, DT) + noise

    # ── Predict (no noise — using nominal control) ──
    pred_state, _     = ekf_predict(pred_state, np.eye(3)*1e-6, V, OMEGA, DT)
    ekf_state, Sigma  = ekf_predict(ekf_state,  Sigma,          V, OMEGA, DT)

    # ── Update with observations from each landmark ──
    for lm in LANDMARKS:
        z_true  = observation_model(true_state, lm)
        z_noisy = z_true + np.random.multivariate_normal([0, 0], R)
        ekf_state, Sigma = ekf_update(ekf_state, Sigma, z_noisy, lm)

    true_traj.append(true_state.copy())
    pred_traj.append(pred_state.copy())
    ekf_traj.append(ekf_state.copy())
    ellipses.append(Sigma.copy())
    sigmas.append(Sigma.copy())

true_traj = np.array(true_traj)
pred_traj = np.array(pred_traj)
ekf_traj  = np.array(ekf_traj)

# ─── Plot 1: Full Trajectory ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(true_traj[:, 0], true_traj[:, 1], 'g-',  lw=2,   label='True Trajectory')
ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', lw=1.5, label='Predicted (Dead Reckoning)')
ax.plot(ekf_traj[:, 0],  ekf_traj[:, 1],  'b-',  lw=2,   label='EKF Corrected Trajectory')

# Plot uncertainty ellipses every 20 steps
for i in range(0, len(ellipses), 20):
    el = get_ellipse(ekf_traj[i+1], ellipses[i])
    ax.add_patch(el)

# Plot landmarks
ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1],
           marker='^', s=150, c='orange', zorder=5, label='Landmarks')
for idx, lm in enumerate(LANDMARKS):
    ax.annotate(f'L{idx+1}', lm, textcoords="offset points",
                xytext=(6, 6), fontsize=9)

ax.scatter(*true_traj[0, :2],  s=120, c='green',  zorder=6, marker='o', label='Start')
ax.scatter(*true_traj[-1, :2], s=120, c='red',    zorder=6, marker='x', label='End')
ax.set_xlabel('X (m)', fontsize=13)
ax.set_ylabel('Y (m)', fontsize=13)
ax.set_title('EKF Localization — Full Trajectory', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/ekf_trajectory.png', dpi=150)
plt.close()
print("Saved: ekf_trajectory.png")

# ─── Plot 2: Position Error Over Time ────────────────────────────────────────
pred_err = np.sqrt((pred_traj[:, 0]-true_traj[:, 0])**2 +
                   (pred_traj[:, 1]-true_traj[:, 1])**2)
ekf_err  = np.sqrt((ekf_traj[:, 0]-true_traj[:, 0])**2 +
                   (ekf_traj[:, 1]-true_traj[:, 1])**2)
time     = np.linspace(0, SIM_TIME, steps+1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, pred_err, 'r--', lw=1.5, label='Dead Reckoning Error')
ax.plot(time, ekf_err,  'b-',  lw=2,   label='EKF Error')
ax.set_xlabel('Time (s)', fontsize=13)
ax.set_ylabel('Position Error (m)', fontsize=13)
ax.set_title('Position Error: Dead Reckoning vs EKF', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/ekf_error.png', dpi=150)
plt.close()
print("Saved: ekf_error.png")

# ─── Plot 3: Uncertainty (trace of covariance) over time ─────────────────────
uncertainty = [np.trace(s[:2,:2]) for s in sigmas]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, uncertainty, 'purple', lw=2)
ax.set_xlabel('Time (s)', fontsize=13)
ax.set_ylabel('Trace of Covariance (m²)', fontsize=13)
ax.set_title('EKF Uncertainty Over Time', fontsize=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/ekf_uncertainty.png', dpi=150)
plt.close()
print("Saved: ekf_uncertainty.png")

# ─── Animation ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('EKF Localization Animation', fontsize=15)
ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1],
           marker='^', s=150, c='orange', zorder=5, label='Landmarks')
for idx, lm in enumerate(LANDMARKS):
    ax.annotate(f'L{idx+1}', lm, textcoords="offset points", xytext=(6,6), fontsize=9)

true_line,  = ax.plot([], [], 'g-',  lw=2,   label='True')
pred_line,  = ax.plot([], [], 'r--', lw=1.5, label='Predicted')
ekf_line,   = ax.plot([], [], 'b-',  lw=2,   label='EKF')
robot_dot,  = ax.plot([], [], 'bo',  ms=8)
ax.legend(fontsize=11, loc='upper right')

ellipse_patch = [None]

def init():
    true_line.set_data([], [])
    pred_line.set_data([], [])
    ekf_line.set_data([], [])
    robot_dot.set_data([], [])
    return true_line, pred_line, ekf_line, robot_dot

def animate(i):
    true_line.set_data(true_traj[:i+1, 0], true_traj[:i+1, 1])
    pred_line.set_data(pred_traj[:i+1, 0], pred_traj[:i+1, 1])
    ekf_line.set_data(ekf_traj[:i+1,  0], ekf_traj[:i+1,  1])
    robot_dot.set_data([ekf_traj[i, 0]], [ekf_traj[i, 1]])

    if ellipse_patch[0] is not None:
        ellipse_patch[0].remove()
    if i > 0:
        el = get_ellipse(ekf_traj[i], ellipses[i-1])
        ax.add_patch(el)
        ellipse_patch[0] = el

    return true_line, pred_line, ekf_line, robot_dot

ani = animation.FuncAnimation(fig, animate, frames=range(0, steps+1, 2),
                               init_func=init, interval=50, blit=False)
ani.save('/home/stranger/Desktop/ekf_animation.mp4',
         writer='ffmpeg', fps=20, dpi=120)
plt.close()
print("Saved: ekf_animation.mp4")

print("\nAll outputs saved successfully!")
