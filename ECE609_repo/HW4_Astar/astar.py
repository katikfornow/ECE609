import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq
import os

# ─── Grid Parameters ──────────────────────────────────────────────────────────
GRID_SIZE   = 20        # 20x20 grid
CELL_SIZE   = 1.0       # each cell = 1m

# Start and Goal
START = (1, 1)
GOAL  = (18, 18)

# Obstacles (x, y) positions
OBSTACLES = [
    # Horizontal walls
    (5,  1), (5,  2), (5,  3), (5,  4), (5,  5), (5,  6),
    (5,  8), (5,  9), (5, 10), (5, 11), (5, 12),
    # Vertical walls
    (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13),
    (10, 15),(10, 16),(10, 17),(10, 18),
    # More obstacles
    (15, 1), (15, 2), (15, 3), (15, 4), (15, 5),
    (15, 8), (15, 9), (15,10), (15,11), (15,12),
    (3,  14),(3,  15),(3,  16),(3,  17),(3,  18),
    (7,  14),(7,  15),(7,  16),(7,  17),
    (12, 1), (12, 2), (12, 3), (12, 4),
]
OBSTACLE_SET = set(OBSTACLES)

# ─── Heuristic (Euclidean distance) ───────────────────────────────────────────
def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# ─── Get valid neighbors (8-connected grid) ────────────────────────────────────
def get_neighbors(node):
    x, y = node
    neighbors = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),
                   (-1,-1),(-1,1),(1,-1),(1,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if (nx, ny) not in OBSTACLE_SET:
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                neighbors.append(((nx, ny), cost))
    return neighbors

# ─── A* Algorithm ─────────────────────────────────────────────────────────────
def astar(start, goal):
    open_set  = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score   = {start: 0}
    f_score   = {start: heuristic(start, goal)}
    visited   = set()
    explored  = []  # for visualization

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)
        explored.append(current)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, explored

        for neighbor, cost in get_neighbors(current):
            if neighbor in visited:
                continue
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f_score[neighbor]   = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, explored  # No path found

# ─── Run A* ───────────────────────────────────────────────────────────────────
print("Running A* search...")
path, explored = astar(START, GOAL)

if path is None:
    print("No path found!")
    exit()

print(f"Path found! Length: {len(path)} steps")
print(f"Explored nodes: {len(explored)}")

# ─── Helper: Draw Grid ────────────────────────────────────────────────────────
def draw_grid(ax, show_explored=True, explored_nodes=None, path=None, robot_pos=None, step=None):
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Obstacles
    for ox, oy in OBSTACLES:
        ax.add_patch(plt.Rectangle((ox-0.5, oy-0.5), 1, 1, color='black'))

    # Explored nodes
    if show_explored and explored_nodes:
        for ex, ey in explored_nodes:
            if (ex, ey) not in OBSTACLE_SET and (ex,ey) != START and (ex,ey) != GOAL:
                ax.add_patch(plt.Rectangle((ex-0.5, ey-0.5), 1, 1,
                             color='lightblue', alpha=0.5))

    # Path
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, 'y-', lw=2.5, zorder=4, label='A* Path')

    # Start and Goal
    ax.add_patch(plt.Rectangle((START[0]-0.5, START[1]-0.5), 1, 1, color='green', zorder=5))
    ax.add_patch(plt.Rectangle((GOAL[0]-0.5,  GOAL[1]-0.5),  1, 1, color='red',   zorder=5))
    ax.text(START[0], START[1], 'S', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold', zorder=6)
    ax.text(GOAL[0],  GOAL[1],  'G', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold', zorder=6)

    # Robot
    if robot_pos:
        ax.plot(robot_pos[0], robot_pos[1], 'bo', ms=12, zorder=7, label='Robot')

    if step is not None:
        ax.set_title(f'A* Path Planning  Step: {step}/{len(path)-1}', fontsize=13)
    else:
        ax.set_title('A* Path Planning Result', fontsize=14)

    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)

# ─── Plot 1: Full A* Result ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9))
draw_grid(ax, show_explored=True, explored_nodes=explored, path=path)
ax.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/astar_result.png', dpi=150)
plt.close()
print("Saved: astar_result.png")

# ─── Plot 2: Path only (clean) ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9))
draw_grid(ax, show_explored=False, path=path)
ax.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/astar_path.png', dpi=150)
plt.close()
print("Saved: astar_path.png")

# ─── Plot 3: Path length and cost ────────────────────────────────────────────
path_lengths = []
cumulative   = 0.0
for i in range(1, len(path)):
    dx = path[i][0] - path[i-1][0]
    dy = path[i][1] - path[i-1][1]
    cumulative += np.sqrt(dx**2 + dy**2)
    path_lengths.append(cumulative)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, len(path)), path_lengths, 'b-', lw=2)
ax.set_xlabel('Path Step', fontsize=12)
ax.set_ylabel('Cumulative Distance (m)', fontsize=12)
ax.set_title(f'A* Path Cost  —  Total: {cumulative:.2f}m  |  Steps: {len(path)}', fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/stranger/Desktop/astar_cost.png', dpi=150)
plt.close()
print("Saved: astar_cost.png")

# ─── Robot Simulation along path ─────────────────────────────────────────────
# Interpolate path for smooth robot motion
smooth_path = []
for i in range(len(path) - 1):
    x0, y0 = path[i]
    x1, y1 = path[i+1]
    for alpha in np.linspace(0, 1, 5, endpoint=False):
        smooth_path.append((x0 + alpha*(x1-x0), y0 + alpha*(y1-y0)))
smooth_path.append(path[-1])

# ─── Save Animation Frames ────────────────────────────────────────────────────
frames_dir = '/home/stranger/Desktop/astar_frames'
os.makedirs(frames_dir, exist_ok=True)

print(f"Saving {len(smooth_path)} animation frames...")
for i, robot_pos in enumerate(smooth_path):
    fig, ax = plt.subplots(figsize=(9, 9))

    # Draw obstacles
    for ox, oy in OBSTACLES:
        ax.add_patch(plt.Rectangle((ox-0.5, oy-0.5), 1, 1, color='black'))

    # Draw explored (light)
    for ex, ey in explored:
        if (ex, ey) not in OBSTACLE_SET and (ex,ey) != START and (ex,ey) != GOAL:
            ax.add_patch(plt.Rectangle((ex-0.5, ey-0.5), 1, 1,
                         color='lightblue', alpha=0.3))

    # Full planned path
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, 'y-', lw=2, alpha=0.6, label='Planned Path')

    # Robot trail
    trail_x = [s[0] for s in smooth_path[:i+1]]
    trail_y = [s[1] for s in smooth_path[:i+1]]
    ax.plot(trail_x, trail_y, 'b-', lw=2, label='Robot Trail')

    # Robot
    ax.plot(robot_pos[0], robot_pos[1], 'bo', ms=12, zorder=7)

    # Start / Goal
    ax.add_patch(plt.Rectangle((START[0]-0.5, START[1]-0.5), 1, 1, color='green', zorder=5))
    ax.add_patch(plt.Rectangle((GOAL[0]-0.5,  GOAL[1]-0.5),  1, 1, color='red',   zorder=5))
    ax.text(START[0], START[1], 'S', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold', zorder=6)
    ax.text(GOAL[0],  GOAL[1],  'G', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold', zorder=6)

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title(f'Robot Following A* Path  —  Step {i+1}/{len(smooth_path)}', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{frames_dir}/frame_{i:04d}.png', dpi=80)
    plt.close()

print("Frames saved!")
print("Now run in terminal:")
print(f'ffmpeg -y -framerate 20 -i "{frames_dir}/frame_%04d.png" -vcodec mpeg4 -q:v 2 /home/stranger/Desktop/astar_animation.avi')
