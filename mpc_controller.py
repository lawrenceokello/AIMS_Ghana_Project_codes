
import cvxpy as cp
import numpy as np
from environment import discrete_dynamics


class MPCController:
    def __init__( self,horizon: int = 15, dt: float = 0.2, u_max: float = 3.5, tree_radius: float = 2.0,
        tree_safety_margin: float = 0.5, inter_drone_min: float = 2.0,boundary: tuple[float, float] = (0.0, 50.0),
        Q_pos: float = 30.0,Q_vel: float = 2.0,R: float = 0.01,slack_weight: float = 1000.0,Q_terminal: float = 5.0,):
        self.N = horizon
        self.dt = dt
        self.u_max = u_max
        self.r_tree = tree_radius
        self.tree_margin = tree_safety_margin
        self.r_tree_safe = tree_radius + tree_safety_margin
        self.d_min = inter_drone_min
        self.x_lo, self.x_hi = boundary
        self.Q_pos = Q_pos
        self.Q_vel = Q_vel
        self.R = R
        self.slack_weight = slack_weight
        self.Q_terminal = Q_terminal

        self.A, self.B = discrete_dynamics(dt)
        self._prob = None  # there is no optimization problem stored yet
        self._n_trees = None #there is no tree

    def _build(self, n_trees: int):
        """Build the QP with time-varying reference trajectory parameters."""
        N = self.N   # prediction horizon 
        A, B = self.A, self.B # discrete matrices        
        self.P_s0 = [cp.Parameter(4) for _ in range(2)] # Initial state parameters in CVPY, 4 parameters for each drone
        self.P_sref_k = [[cp.Parameter(4) for _ in range(N)] for _ in range(2)] # Time-varying reference trajectory:[x_ref,y_ref,vx_ref,vy_ref]
        self.P_tree_n = [cp.Parameter((n_trees, 2)) for _ in range(2)]  # for each tree [nx,ny] linearizes obstacle constraints
        self.P_tree_rhs = [cp.Parameter(n_trees) for _ in range(2)] # safe distance from the tree
        self.P_drone_n = cp.Parameter(2) ## drone separation distance

        # Decision variables
        self.V_s = [cp.Variable((4, N + 1)) for _ in range(2)]  # store the state variables
        self.V_u = [cp.Variable((2, N)) for _ in range(2)]  # stores the control variables
        self.V_eps_tree = [cp.Variable((n_trees, N), nonneg=True) for _ in range(2)] # slack variables 
        self.V_eps_drone = cp.Variable(N, nonneg=True) # slack for the drones 

        cost = 0
        cons = []
        for i in range(2):
            # Initial state constraint
            cons.append(self.V_s[i][:, 0] == self.P_s0[i]) ## defining the initial state constraint
            for k in range(N):
                cons.append(self.V_s[i][:, k + 1] == A @ self.V_s[i][:, k] + B @ self.V_u[i][:, k])                
                cons.append(cp.abs(self.V_u[i][:, k]) <= self.u_max) # Control limits                
                cons.append(self.V_s[i][0:2, k + 1] >= self.x_lo + 0.1) # Boundary constraints
                cons.append(self.V_s[i][0:2, k + 1] <= self.x_lu - 0.1) # Boundary constraints
                is_terminal = 1.0 if k == N - 1 else 0.0   # Tracking cost with time-varying reference
                q_mult = 1.0 + (self.Q_terminal - 1.0) * is_terminal  # terminal multiplier for smooth ending at N-1
                err_pos = self.V_s[i][0:2, k + 1] - self.P_sref_k[i][k][0:2] # error in the position
                err_vel = self.V_s[i][2:4, k + 1] - self.P_sref_k[i][k][2:4] # error in th velocity
                cost += q_mult * self.Q_pos * cp.sum_squares(err_pos)
                cost += q_mult * self.Q_vel * cp.sum_squares(err_vel)
                cost += self.R * cp.sum_squares(self.V_u[i][:, k])

            # Tree avoidance constraints (linearized half-plane)
            for j in range(n_trees):
                for k in range(N):
                    cons.append(self.P_tree_n[i][j, :] @ self.V_s[i][0:2, k + 1] + self.V_eps_tree[i][j, k] >= self.P_tree_rhs[i][j] )
                # Slack cost (high weight + quadratic prevents constraint violation)
                cost += self.slack_weight * cp.sum(self.V_eps_tree[i][j, :])
                cost += 10.0 * cp.sum_squares(self.V_eps_tree[i][j, :])        
        for k in range(N):  # interdrone separation
            cons.append(self.P_drone_n @ (self.V_s[0][0:2, k + 1] - self.V_s[1][0:2, k + 1])+ self.V_eps_drone[k] >= self.d_min )
        cost += self.slack_weight * cp.sum(self.V_eps_drone)
        cost += 10.0 * cp.sum_squares(self.V_eps_drone)
        self._prob = cp.Problem(cp.Minimize(cost), cons)
        self._n_trees = n_trees

    def solve(self, s0: np.ndarray, s_ref_traj: np.ndarray, trees: np.ndarray):
        n_trees = len(trees)
        if self._prob is None or self._n_trees != n_trees:
            self._build(n_trees)
        for i in range(2):
            self.P_s0[i].value = s0[i]
            for k in range(self.N):
                if k < s_ref_traj.shape[1]:
                    self.P_sref_k[i][k].value = s_ref_traj[i, k]
                else:
                    self.P_sref_k[i][k].value = s_ref_traj[i, -1]
            normals = np.zeros((n_trees, 2))  # Compute linearized tree avoidance constraints
            rhs = np.zeros(n_trees)
            p_i = s0[i, 0:2]
            v_i = s0[i, 2:4]
            p_mid = p_i + 0.5 * v_i * self.dt * self.N  ## determines the position of the drone
            for j, tree in enumerate(trees):
                diff = p_mid - tree
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = p_i - tree
                    dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = np.array([1.0, 0.0]); dist = 1.0
                n = diff / dist
                normals[j] = n
                rhs[j] = self.r_tree_safe + n @ tree
            self.P_tree_n[i].value = normals
            self.P_tree_rhs[i].value = rhs

        # Inter-drone separation direction
        diff_d = s0[0, 0:2] - s0[1, 0:2]
        d_now = np.linalg.norm(diff_d)
        if d_now < 1e-6:
            diff_d = np.array([1.0, 0.0]); d_now = 1.0
        self.P_drone_n.value = diff_d / d_now

        try:
            self._prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=15000, eps_abs=1e-5, eps_rel=1e-5,
                             polish=True, adaptive_rho=True)
            if self._prob.status not in ("optimal", "optimal_inaccurate"):
                return np.zeros((2, 2)), False
        except Exception:
            return np.zeros((2, 2)), False

        u0 = np.stack([self.V_u[0].value[:, 0], self.V_u[1].value[:, 0]])
        return u0, True
