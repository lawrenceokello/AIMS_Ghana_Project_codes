
import numpy as np


# ============================================================================
# Drone dynamics
# ============================================================================

def discrete_dynamics(dt: float):
    A = np.array([[1, 0, dt, 0],[0, 1, 0, dt],[0, 0, 1, 0],[0, 0, 0, 1], ], dtype=float)
    B = np.array([[0.5 * dt * dt, 0],[0, 0.5 * dt * dt],[dt, 0], [0, dt],], dtype=float)
    return A, B
# Boustrophedon waypoint generator

def generate_boustrophedon_waypoints(x_lo, x_lu, y_lo, y_lu,D_x=4, y_step=1.5):
    wp = []
    x_positions = np.arange(x_lo, x_hi, D_x)
    for i, x in enumerate(x_positions):
        if i % 2 == 0:
            y_positions = np.arange(y_lo, y_lu, D_y)
        else:
            y_positions = np.arange(y_lu, y_lo, -D_y)
        for y in y_positions:
            wp.append([x, y])
    return np.array(wp, dtype=float)

def generate_ideal_lanes(x_lo, x_lu, y_lo, y_lu, D_x,):
    lanes = []
    x_po = np.arange(x_lo + D_x / 2, x_lu, D_x)
    for x in x_po:
        lanes.append(np.array([[x, y_lo + 0.5], [x, y_lu - 0.5]]))
    return lanes



# Environment

class GardenEnvironment:

    def __init__(self, size: float = 50.0, n_trees_per_region: int = 4, grid_res: int = 20,spray_radius: float = 2.5,
        tree_radius: float = 1.5, inter_drone: float = 2.0, max_steps: int = 600, dt: float = 0.1, max_speed: float = 5.0, 
        max_accel: float = 2.0, D_x: float = 4.0, D_y: float = 1.5, wp_reach_dist: float = 2.0):
        self.size = size
        self.n_trees_per_region = n_trees_per_region
        self.grid_res = grid_res
        self.spray_radius = spray_radius
        self.tree_radius = tree_radius
        self.inter_drone_min = inter_drone
        self.max_steps = max_steps
        self.dt = dt
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.lane_spacing = D_x
        self.y_step = D_y
        self.wp_reach_dist = wp_reach_dist
        self.A, self.B = discrete_dynamics(dt)
        self.obs_dim = 11
        self.act_dim = 2
        self.n_agents = 2

        # Cell centres for the coverage grid
        edges = np.linspace(0, size, grid_res + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        cx, cy = np.meshgrid(centres, centres, indexing="xy")
        self.cell_centres = np.stack([cx.ravel(), cy.ravel()], axis=1)

        # Region assignment: drone 0 -> left half (x < size/2), drone 1 -> right
        self.cell_region = (self.cell_centres[:, 0] >= size / 2).astype(int)

        # Placeholders, set in reset()
        self.trees = None
        self.coverage = None
        self.states = None
        self.waypoints = None
        self.wp_idx = None
        self.ideal_lanes = None 
        self.step_count = 0
        self.done = False

    # ------------------------------------------------------------------
    # Boustrophedon waypoints and ideal lanes
    # ------------------------------------------------------------------

    def _generate_waypoints(self):
        half = self.size / 2
        margin = 1.5
        wps = []
        for i in range(self.n_agents):
            x_lo = margin if i == 0 else half + 0.5
            x_lu = half - 0.5 if i == 0 else self.size - margin
            wps.append(generate_boustrophedon_waypoints(
                x_lo=x_lo, x_lu=x_lu,
                y_lo=0, y_hi=self.size,
                D_x=self.D_x,
                y_step=self.D_y,
            ))
        return wps

    def _generate_ideal_lanes(self):
        """Generate ideal straight vertical lane lines for each drone's region."""
        half = self.size / 2
        margin = 1.5
        lanes = []
        for i in range(self.n_agents):
            x_lo = margin if i == 0 else half + 0.5
            x_hi = half - 0.5 if i == 0 else self.size - margin
            lanes.append(generate_ideal_lanes(
                x_lo=x_lo, x_lu=x_lu,
                y_lo=0, y_hi=self.size,
                D_x=self.D_x,
            ))
        return lanes

    def _advance_waypoints(self):
        """Advance waypoint index if drone is close enough to current target."""
        for i in range(self.n_agents):
            if self.wp_idx[i] < len(self.waypoints[i]) - 1:
                pos = self.states[i, :2]
                wp = self.waypoints[i][self.wp_idx[i]]
                if np.linalg.norm(pos - wp) < self.wp_reach_dist:
                    self.wp_idx[i] += 1

    def _current_waypoint(self, drone_idx: int):
        """Get the current target waypoint for a drone."""
        return self.waypoints[drone_idx][self.wp_idx[drone_idx]]

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Place trees, reset drones at first waypoint, generate waypoints and lanes."""
        self.trees = self._sample_trees()
        self.coverage = np.zeros(self.grid_res ** 2, dtype=bool)

        # Generate boustrophedon waypoints and ideal lane lines
        self.waypoints = self._generate_waypoints()
        self.ideal_lanes = self._generate_ideal_lanes()
        self.wp_idx = [0, 0]

        # Start each drone at its first waypoint (bottom of first column)
        starts = []
        for i in range(self.n_agents):
            wp0 = self.waypoints[i][0]
            starts.append([wp0[0], wp0[1], 0.0, 0.0])
        self.states = np.array(starts)

        self.step_count = 0
        self.done = False
        self._prev_dist = [None, None]
        self._prev_wp_dist = [None, None]
        return self._get_observations()

    def step(self, actions: np.ndarray):
        """Advance one step. `actions` is (2, 2): per-drone acceleration in m/s^2."""
        actions = np.clip(actions, -self.max_accel, self.max_accel)

        for i in range(self.n_agents):
            self.states[i] = self.A @ self.states[i] + self.B @ actions[i]
            v = self.states[i, 2:4]
            speed = np.linalg.norm(v)
            if speed > self.max_speed:
                self.states[i, 2:4] = v * (self.max_speed / speed)

        self._advance_waypoints()
        self.step_count += 1
        rewards, info = self._compute_rewards(actions)

        all_sprayed = self.coverage.all()
        out_of_time = self.step_count >= self.max_steps
        self.done = bool(all_sprayed or out_of_time)
        info["all_sprayed"] = bool(all_sprayed)
        info["timeout"] = bool(out_of_time)

        return self._get_observations(), rewards, self.done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_trees(self):
        """Place trees randomly in each region."""
        trees = []
        margin = 5.0
        for region in (0, 1):
            x_lo = 0 + margin if region == 0 else self.size / 2 + 2.0
            x_hi = self.size / 2 - 2.0 if region == 0 else self.size - margin
            for _ in range(self.n_trees_per_region):
                for _attempt in range(50):
                    x = self.rng.uniform(x_lo, x_hi)
                    y = self.rng.uniform(margin, self.size - margin)
                    if all(np.hypot(x - tx, y - ty) > 4.0 for tx, ty in trees):
                        trees.append((x, y))
                        break
        return np.array(trees)

    def _get_observations(self):
        obs = np.zeros((self.n_agents, self.obs_dim))
        for i in range(self.n_agents):
            j = 1 - i
            pos_i = self.states[i, :2]
            pos_j = self.states[j, :2]

            # Current waypoint direction (normalized relative vector)
            wp = self._current_waypoint(i)
            wp_rel = wp - pos_i
            wp_dist = np.linalg.norm(wp_rel)
            if wp_dist > 1e-6:
                wp_dir = wp_rel / self.size
            else:
                wp_dir = np.zeros(2)

            # Waypoint progress
            wp_progress = self.wp_idx[i] / max(len(self.waypoints[i]) - 1, 1)

            # nearest tree
            if len(self.trees) > 0:
                dtrees = self.trees - pos_i
                dtree_dists = np.linalg.norm(dtrees, axis=1)
                idx_t = np.argmin(dtree_dists)
                near_tree = dtrees[idx_t]
                near_tree_d = dtree_dists[idx_t] / self.size
            else:
                near_tree = np.array([self.size, self.size])
                near_tree_d = 1.0

            cov_frac = self.coverage[self.cell_region == i].mean()

            obs[i] = np.array([
                pos_i[0] / self.size, pos_i[1] / self.size,
                self.states[i, 2] / self.max_speed,
                self.states[i, 3] / self.max_speed,
                wp_dir[0],
                wp_dir[1],
                wp_progress,
                near_tree[0] / self.size,
                near_tree[1] / self.size,
                near_tree_d,
                (pos_j[0] - pos_i[0]) / self.size,
                (pos_j[1] - pos_i[1]) / self.size,
                cov_frac,
            ])
        return obs

    def _compute_rewards(self, actions):
        rewards = np.zeros(self.n_agents)

        # 1) Waypoint progression reward — strong signal to follow boustrophedon
        for i in range(self.n_agents):
            pos = self.states[i, :2]
            wp = self._current_waypoint(i)
            d_now = float(np.linalg.norm(wp - pos))
            if self._prev_wp_dist is not None and self._prev_wp_dist[i] is not None:
                rewards[i] += 0.5 * (self._prev_wp_dist[i] - d_now)
            self._prev_wp_dist[i] = d_now
            if d_now < self.wp_reach_dist:
                rewards[i] += 2.0

        # 2) Potential-based shaping toward nearest unsprayed cell in own region
        if not hasattr(self, "_prev_dist") or self._prev_dist is None:
            self._prev_dist = [None, None]
        for i in range(self.n_agents):
            pos = self.states[i, :2]
            mask = (self.cell_region == i) & (~self.coverage)
            if mask.any():
                cells = self.cell_centres[mask]
                d_now = float(np.linalg.norm(cells - pos, axis=1).min())
                if self._prev_dist[i] is not None:
                    rewards[i] += 0.3 * (self._prev_dist[i] - d_now)
                self._prev_dist[i] = d_now
            else:
                self._prev_dist[i] = 0.0

        # 3) Coverage reward
        prev_cov = self.coverage.copy()
        for i in range(self.n_agents):
            pos = self.states[i, :2]
            d = np.linalg.norm(self.cell_centres - pos, axis=1)
            in_region = (self.cell_region == i)
            newly = (d < self.spray_radius) & in_region & (~self.coverage)
            self.coverage |= newly
        gain = int(self.coverage.sum() - prev_cov.sum())
        info["coverage_gain"] = gain
        rewards += 5.0 * gain

        # 4) Tree contact penalty
        for i in range(self.n_agents):
            pos = self.states[i, :2]
            dtrees_vec = pos - self.trees
            dtrees = np.linalg.norm(dtrees_vec, axis=1)
            min_idx = int(np.argmin(dtrees))
            min_d = float(dtrees[min_idx])
            if min_d < self.tree_radius:
                rewards[i] -= 10.0
                if min_d > 1e-6:
                    n_hat = dtrees_vec[min_idx] / min_d
                else:
                    n_hat = np.array([1.0, 0.0])
                self.states[i, 0:2] = self.trees[min_idx] + n_hat * (self.tree_radius + 0.1)
                self.states[i, 2:4] *= 0.0
            else:
                close = self.tree_radius + 1.5
                if min_d < close:
                    rewards[i] -= 0.6 * (close - min_d)

        # 5) Inter-drone separation
        d_inter = np.linalg.norm(self.states[0, :2] - self.states[1, :2])
        if d_inter < self.inter_drone_min and d_inter > 1e-6:
            rewards -= 3.0
            n_d = (self.states[0, :2] - self.states[1, :2]) / d_inter
            push = 0.5 * (self.inter_drone_min - d_inter)
            self.states[0, 0:2] += n_d * push
            self.states[1, 0:2] -= n_d * push
            self.states[0, 2:4] *= 0.0
            self.states[1, 2:4] *= 0.0

        # 6) Boundary
        for i in range(self.n_agents):
            for d in (0, 1):
                if self.states[i, d] < 0:
                    self.states[i, d] = 0.0
                    self.states[i, d + 2] = 0.0
                    rewards[i] -= 0.5
                elif self.states[i, d] > self.size:
                    self.states[i, d] = self.size
                    self.states[i, d + 2] = 0.0
                    rewards[i] -= 0.5

        # 7) Step penalty and control-effort
        rewards -= 0.01
        rewards -= 0.003 * np.sum(actions ** 2, axis=1)

        return rewards, info

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def positions(self):
        return self.states[:, :2].copy()

    def coverage_fraction(self):
        return float(self.coverage.mean())
