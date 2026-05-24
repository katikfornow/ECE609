import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

np.random.seed(42)

# ─── Parameters ──────────────────────────────────────────────────────────────
DT       = 0.1
SIM_TIME = 20.0
V        = 1.0
OMEGA    = 0.314
N_PARTICLES = 1000

# Motion noise
R_STD  = np.array([0.1, 0.1, np.deg2rad(2.0)])  # x, y, theta std

# Observation noise
Q_STD  = np.array([0.2, np.deg2rad(5.0)])         # range, bearing std

# Landmarks
LANDMARKS = np.array([
    [4.0,  0.0], [3.0,  3.0], [0.0,  4.0], [-3.0, 3.0],
    [-4.0, 0.0], [-3.0,-3.0], [0.0, -4.0], [3.0, -3.0],
])

# ─── Motion Model (Unicycle) ──────────────────────────────────────────────────
def motion_model(state, v, omega, dt):
    x, y, theta = state
    x     += v * np.cos(theta) * dt
    y     += v * np.sin(theta) * dt
    theta += omega * dt
    return np.array([x, y, theta])

def sample_motion_model(particles, v, omega, dt):
    """Apply motion model with noise to all particles"""
    n = len(particles)
    noise = np.random.randn(n, 3) * R_STD
    new_particles = np.zeros_like(particles)
    for i in range(n):
        new_particles[i] = motion_model(particles[i], v, omega, dt) + noise[i]
    return new_particles

# ─── Observation Model (Range & Bearing) ─────────────────────────────────────
def observation_model(state, landmark):
    dx  = landmark[0] - state[0]
    dy  = landmark[1] - state[1]
    r   = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx) - state[2]
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    return np.array([r, phi])

def compute_weight(particle, observations):
    """Compute particle weight based on how well it explains observations"""
    weight = 1.0
    for z, lm in observations:
        z_hat = observation_model(particle, lm)
        diff  = z - z_hat
        diff[1] = np.arctan2(np.sin(diff[1]), np.cos(diff[1]))
        # Gaussian likelihood
        exponent = -0.5 * (
            (diff[0]/Q_STD[0])**2 +
            (diff[1]/Q_STD[1])**2
        )
        weight *= np.exp(exponent) + 1e-300  # avoid zero
    return weight

# ─── Low Variance Resampling ──────────────────────────────────────────────────
def low_variance_resample(particles, weights):
    n      = len(particles)
    new_p  = np.zeros_like(particles)
    r      = np.random.uniform(0, 1.0/n)
    c      = weights[0]
    i      = 0
    for m in range(n):
        u = r + m * (1.0/n)
        while u > c and i < n-1:
            i += 1
            c += weights[i]
        new_p[m] = particles[i]
    return new_p

# ─── Estimate pose from particles ─────────────────────────────────────────────
def estimate_pose(particles, weights):
    x   = np.average(particles[:, 0], weights=weights)
    y   = np.average(particles[:, 1], weights=weights)
    sin_sum = np.average(np.sin(particles[:, 2]), weights=weights)
    cos_sum = np.average(np.cos(particles[:, 2]), weights=weights)
    theta = np.arctan2(sin_sum, cos_sum)
    return np.array([x, y, theta])

# ─── Run Simulation ───────────────────────────────────────────────────────────
steps = int(SIM_TIME / DT)

# Initialize true state
true_state = np.array([0.0, 0.0, 0.0])

# Initialize particles uniformly around start
particles = np.zeros((N_PARTICLES, 3))
particles[:, 0] = np.random.uniform(-1.0, 1.0, N_PARTICLES)
particles[:, 1] = np.random.uniform(-1.0, 1.0, N_PARTICLES)
particles[:, 2] = np.random.uniform(-np.pi, np.pi, N_PARTICLES)
weights = np.ones(N_PARTICLES) / N_PARTICLES

# Storage
true_traj      = [true_state.copy()]
est_traj       = [estimate_pose(particles, weights)]
all_particles  = [particles.copy()]
all_weights    = [weights.copy()]
all_obs        = []

print("Running MCL simulation...")
for step in range(steps):
    # ── True state update ──
    noise      = np.random.randn(3) * R_STD * 0.5
    true_state = motion_model(true_state, V, OMEGA, DT) + noise

    # ── Get noisy observations from all landmarks ──
    observations = []
    for lm in LANDMARKS:
        z_true  = observation_model(true_state, lm)
        z_noisy = z_true + np.random.randn(2) * Q_STD
        observations.append((z_noisy, lm))

    # ── MCL Steps ──
    # 1. Sample from motion model
    particles = sample_motion_model(particles, V, OMEGA, DT)

    # 2. Compute weights
    weights = np.array([compute_weight(p, observations) for p in particles])
    weights += 1e-300
    weights /= weights.sum()

    # 3. Resample
    particles = low_variance_resample(particles, weights)
    weights   = np.ones(N_PARTICLES) / N_PARTICLES

    # ── Store ──
    true_traj.append(true_state.copy())
    est_traj.append(estimate_pose(particles, weights))
    all_particles.append(particles.copy())
    all_weights.append(weights.copy())
    all_obs.append(observations)

true_traj = np.array(true_traj)
est_traj  = np.array(est_traj)
print("Simulation done!")

# ─── Plot 1: Full Trajectory ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))

# Plot particles at a few steps
for i in [50, 100, 150]:
    ax.scatter(all_particles[i][:, 0], all_particles[i][:, 1],
               s=2, c='gray', alpha=0.3)

ax.plot(true_traj[:, 0], true_traj[:, 1], 'g-',  lw=2,   label='True Trajectory')
ax.plot(est_traj[:, 0],  est_traj[:, 1],  'b--', lw=2,   label='MCL Estimate')
ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1],
           marker='^', s=150, c='orange', zorder=5, label='Landmarks')
for i, lm in enumerate(LANDMARKS):
    ax.annotate(f'L{i+1}', lm, xytext=(6,6), textcoords='offset points', fontsize=9)

ax.scatter(*true_traj[0, :2],  s=120, c='green', zorder=6, marker='o', label='Start')
ax.scatter(*true_traj[-1, :2], s=120, c='red',   zorder=6, marker='x', label='End')
ax.set_xlabel('X (m)', fontsize=13)
ax.set_ylabel('Y (m)', fontsize=13)
ax.set_title('MCL Localization — Full Trajectory', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/mcl_trajectory.png', dpi=150)
plt.close()
print("Saved: mcl_trajectory.png")

# ─── Plot 2: Position Error Over Time ────────────────────────────────────────
err = np.sqrt((est_traj[:, 0]-true_traj[:, 0])**2 +
              (est_traj[:, 1]-true_traj[:, 1])**2)
time = np.linspace(0, SIM_TIME, steps+1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, err, 'b-', lw=2, label='MCL Position Error')
ax.axhline(y=np.mean(err), color='r', linestyle='--', label=f'Mean Error: {np.mean(err):.3f}m')
ax.set_xlabel('Time (s)', fontsize=13)
ax.set_ylabel('Position Error (m)', fontsize=13)
ax.set_title('MCL Position Error Over Time', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/mcl_error.png', dpi=150)
plt.close()
print("Saved: mcl_error.png")

# ─── Plot 3: Particle spread over time ───────────────────────────────────────
spread = [np.std(all_particles[i][:, :2]) for i in range(len(all_particles))]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, spread, 'purple', lw=2)
ax.set_xlabel('Time (s)', fontsize=13)
ax.set_ylabel('Particle Spread (std, m)', fontsize=13)
ax.set_title('MCL Particle Spread Over Time', fontsize=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/mcl_spread.png', dpi=150)
plt.close()
print("Saved: mcl_spread.png")

# ─── Save Animation Frames ────────────────────────────────────────────────────
frames_dir = '/home/stranger/Desktop/mcl_frames'
os.makedirs(frames_dir, exist_ok=True)

print("Saving animation frames...")
for i in range(0, steps+1, 2):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'MCL Localization  t={i*DT:.1f}s  N={N_PARTICLES} particles', fontsize=12)

    # Particles
    ax.scatter(all_particles[i][:, 0], all_particles[i][:, 1],
               s=3, c='gray', alpha=0.4, label='Particles')

    # Trajectories
    ax.plot(true_traj[:i+1, 0], true_traj[:i+1, 1], 'g-',  lw=2, label='True')
    ax.plot(est_traj[:i+1,  0], est_traj[:i+1,  1], 'b--', lw=2, label='MCL Estimate')

    # Landmarks
    ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1],
               marker='^', s=150, c='orange', zorder=5)
    for j, lm in enumerate(LANDMARKS):
        ax.annotate(f'L{j+1}', lm, xytext=(5,5),
                    textcoords='offset points', fontsize=8)

    # Observations at current step
    if i > 0:
        for z, lm in all_obs[i-1]:
            ax.plot([true_traj[i,0], lm[0]], [true_traj[i,1], lm[1]],
                    'r-', alpha=0.3, lw=0.8)

    # Current true pose
    ax.plot(true_traj[i, 0], true_traj[i, 1], 'go', ms=10)
    # Current estimate
    ax.plot(est_traj[i, 0],  est_traj[i, 1],  'b*', ms=12)

    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{frames_dir}/frame_{i//2:04d}.png', dpi=80)
    plt.close()

print(f"Frames saved to {frames_dir}")
print("Now run this command to create video:")
print(f'ffmpeg -y -framerate 20 -i "{frames_dir}/frame_%04d.png" -vcodec mpeg4 -q:v 2 /home/stranger/Desktop/mcl_animation.avi')
