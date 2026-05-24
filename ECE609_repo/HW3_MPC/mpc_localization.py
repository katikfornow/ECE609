import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# ─── Parameters ──────────────────────────────────────────────────────────────
DT          = 0.1       # time step (s)
SIM_TIME    = 30.0      # total simulation time (s)
HORIZON     = 10        # MPC prediction horizon (steps)

# Robot constraints
MAX_V       =  1.0      # max linear velocity (m/s)
MIN_V       = -0.5      # min linear velocity (m/s)
MAX_OMEGA   =  1.0      # max angular velocity (rad/s)
MIN_OMEGA   = -1.0      # min angular velocity (rad/s)

# Cost weights
W_POS       = 10.0      # position error weight
W_THETA     = 1.0       # heading error weight
W_V         = 0.5       # linear velocity cost
W_OMEGA     = 1.0       # angular velocity cost
W_Domega     = 2.0      # change in omega (smoothness)

# ─── Reference Trajectory (Square Loop with sharp turns) ─────────────────────
def generate_square_reference(sim_time, dt):
    """Generate a square loop reference trajectory"""
    steps     = int(sim_time / dt)
    ref       = np.zeros((steps, 3))  # [x, y, theta]

    # Square waypoints
    waypoints = [
        (0.0,  0.0,  0.0),
        (4.0,  0.0,  0.0),
        (4.0,  4.0,  np.pi/2),
        (0.0,  4.0,  np.pi),
        (0.0,  0.0, -np.pi/2),
        (4.0,  0.0,  0.0),
        (4.0,  4.0,  np.pi/2),
        (0.0,  4.0,  np.pi),
        (0.0,  0.0, -np.pi/2),
    ]

    seg_steps = steps // (len(waypoints) - 1)

    for seg in range(len(waypoints) - 1):
        x0, y0, t0 = waypoints[seg]
        x1, y1, t1 = waypoints[seg+1]
        start = seg * seg_steps
        end   = min(start + seg_steps, steps)
        for i, idx in enumerate(range(start, end)):
            alpha     = i / seg_steps
            ref[idx, 0] = x0 + alpha * (x1 - x0)
            ref[idx, 1] = y0 + alpha * (y1 - y0)
            ref[idx, 2] = t0 + alpha * (t1 - t0)

    return ref

# ─── Unicycle Motion Model ─────────────────────────────────────────────────────
def unicycle_model(state, v, omega, dt):
    x, y, theta = state
    x     += v * np.cos(theta) * dt
    y     += v * np.sin(theta) * dt
    theta += omega * dt
    theta  = np.arctan2(np.sin(theta), np.cos(theta))
    return np.array([x, y, theta])

# ─── MPC Cost Function ────────────────────────────────────────────────────────
def mpc_cost(u_flat, state, ref_traj, horizon, dt):
    """Compute total MPC cost over prediction horizon"""
    u    = u_flat.reshape(horizon, 2)  # [v, omega] for each step
    cost = 0.0
    s    = state.copy()

    for k in range(horizon):
        v     = np.clip(u[k, 0], MIN_V, MAX_V)
        omega = np.clip(u[k, 1], MIN_OMEGA, MAX_OMEGA)

        s = unicycle_model(s, v, omega, dt)

        ref = ref_traj[k]

        # Position + heading error
        dx    = s[0] - ref[0]
        dy    = s[1] - ref[1]
        dth   = np.arctan2(np.sin(s[2] - ref[2]), np.cos(s[2] - ref[2]))

        cost += W_POS   * (dx**2 + dy**2)
        cost += W_THETA * dth**2
        cost += W_V     * v**2
        cost += W_OMEGA * omega**2

        # Smoothness
        if k > 0:
            cost += W_Domega * (u[k, 1] - u[k-1, 1])**2

    return cost

# ─── Simple MPC using gradient-free optimization (random shooting) ────────────
def mpc_solve(state, ref_traj, horizon, dt, prev_u=None):
    """
    Random shooting MPC:
    Sample many control sequences, pick the best one.
    Simple but effective for demonstration.
    """
    N_SAMPLES = 500
    best_cost = float('inf')
    best_u    = np.zeros((horizon, 2))

    # Warm start from previous solution
    if prev_u is not None:
        warm = prev_u.copy()
    else:
        warm = np.zeros((horizon, 2))

    # Sample random control sequences around warm start
    for _ in range(N_SAMPLES):
        noise = np.random.randn(horizon, 2) * np.array([0.3, 0.3])
        u_sample = warm + noise
        u_sample[:, 0] = np.clip(u_sample[:, 0], MIN_V,     MAX_V)
        u_sample[:, 1] = np.clip(u_sample[:, 1], MIN_OMEGA, MAX_OMEGA)

        cost = mpc_cost(u_sample.flatten(), state, ref_traj, horizon, dt)
        if cost < best_cost:
            best_cost = cost
            best_u    = u_sample.copy()

    return best_u

# ─── Run Simulation ───────────────────────────────────────────────────────────
steps     = int(SIM_TIME / DT)
ref_traj  = generate_square_reference(SIM_TIME, DT)

state     = np.array([0.0, 0.0, 0.0])
prev_u    = None

actual_traj = [state.copy()]
pred_trajs  = []
v_log       = []
omega_log   = []

print("Running MPC simulation...")
for t in range(steps):
    # Get reference for next HORIZON steps
    end_idx  = min(t + HORIZON, steps)
    ref_win  = ref_traj[t:end_idx]

    # Pad if near end
    if len(ref_win) < HORIZON:
        pad     = np.tile(ref_win[-1], (HORIZON - len(ref_win), 1))
        ref_win = np.vstack([ref_win, pad])

    # Solve MPC
    best_u = mpc_solve(state, ref_win, HORIZON, DT, prev_u)

    # Apply first control action
    v     = np.clip(best_u[0, 0], MIN_V,     MAX_V)
    omega = np.clip(best_u[0, 1], MIN_OMEGA, MAX_OMEGA)

    state = unicycle_model(state, v, omega, DT)

    # Shift control sequence for warm start
    prev_u    = np.vstack([best_u[1:], best_u[-1:]])

    actual_traj.append(state.copy())
    v_log.append(v)
    omega_log.append(omega)

    if t % 50 == 0:
        print(f"  Step {t}/{steps}")

actual_traj = np.array(actual_traj)
print("Simulation done!")

# ─── Plot 1: Trajectory ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(ref_traj[:, 0],    ref_traj[:, 1],    'r--', lw=2,   label='Reference')
ax.plot(actual_traj[:, 0], actual_traj[:, 1], 'b-',  lw=2,   label='MPC Output')
ax.scatter(*actual_traj[0, :2],  s=120, c='green', zorder=6, marker='o', label='Start')
ax.scatter(*actual_traj[-1, :2], s=120, c='red',   zorder=6, marker='x', label='End')

# Mark corners
corners = [(0,0), (4,0), (4,4), (0,4)]
for cx, cy in corners:
    ax.plot(cx, cy, 'k^', ms=10, zorder=5)

ax.set_xlabel('X (m)', fontsize=13)
ax.set_ylabel('Y (m)', fontsize=13)
ax.set_title('MPC Trajectory Tracking — Square Loop', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/mpc_trajectory.png', dpi=150)
plt.close()
print("Saved: mpc_trajectory.png")

# ─── Plot 2: Position Error ───────────────────────────────────────────────────
err = np.sqrt((actual_traj[:-1, 0] - ref_traj[:, 0])**2 +
              (actual_traj[:-1, 1] - ref_traj[:, 1])**2)
time = np.linspace(0, SIM_TIME, steps)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, err, 'b-', lw=2, label='Position Error')
ax.axhline(y=np.mean(err), color='r', linestyle='--',
           label=f'Mean: {np.mean(err):.3f}m')
ax.set_xlabel('Time (s)', fontsize=13)
ax.set_ylabel('Position Error (m)', fontsize=13)
ax.set_title('MPC Position Tracking Error', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/mpc_error.png', dpi=150)
plt.close()
print("Saved: mpc_error.png")

# ─── Plot 3: Control Inputs ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(time, v_log,     'g-', lw=2, label='Linear Velocity v')
ax1.axhline(y=MAX_V,     color='r', linestyle='--', alpha=0.5, label='Max v')
ax1.axhline(y=MIN_V,     color='r', linestyle='--', alpha=0.5, label='Min v')
ax1.set_ylabel('v (m/s)', fontsize=12)
ax1.set_title('MPC Control Inputs', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(time, omega_log, 'purple', lw=2, label='Angular Velocity ω')
ax2.axhline(y=MAX_OMEGA,  color='r', linestyle='--', alpha=0.5, label='Max ω')
ax2.axhline(y=MIN_OMEGA,  color='r', linestyle='--', alpha=0.5, label='Min ω')
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('ω (rad/s)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/mpc_controls.png', dpi=150)
plt.close()
print("Saved: mpc_controls.png")

# ─── Save Animation Frames ────────────────────────────────────────────────────
frames_dir = '/home/stranger/Desktop/mpc_frames'
os.makedirs(frames_dir, exist_ok=True)

print("Saving animation frames...")
for i in range(0, steps, 3):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, 5.5)
    ax.set_ylim(-1, 5.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'MPC Trajectory Tracking  t={i*DT:.1f}s', fontsize=13)

    # Reference
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'r--', lw=1.5, alpha=0.5, label='Reference')

    # Actual so far
    ax.plot(actual_traj[:i+1, 0], actual_traj[:i+1, 1], 'b-', lw=2, label='MPC')

    # Robot position
    x, y, theta = actual_traj[i]
    ax.plot(x, y, 'bo', ms=10)
    ax.annotate('', xy=(x + 0.3*np.cos(theta), y + 0.3*np.sin(theta)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # Corners
    for cx, cy in [(0,0),(4,0),(4,4),(0,4)]:
        ax.plot(cx, cy, 'k^', ms=10, zorder=5)

    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{frames_dir}/frame_{i//3:04d}.png', dpi=80)
    plt.close()

print(f"Frames saved!")
print("Now run in terminal:")
print(f'ffmpeg -y -framerate 20 -i "{frames_dir}/frame_%04d.png" -vcodec mpeg4 -q:v 2 /home/stranger/Desktop/mpc_animation.avi')
