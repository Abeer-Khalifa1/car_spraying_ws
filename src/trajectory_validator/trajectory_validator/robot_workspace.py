import numpy as np
from scipy.spatial.transform import Rotation


JOINTS: list[dict] = [
    {'xyz': np.array([0.0000,  0.0000,  0.1058]), 'rpy': np.array([0.0,    0.0,     0.0   ]), 'axis': np.array([0, 0, 1])},  # joint_0
    {'xyz': np.array([0.0550,  0.0000,  0.0660]), 'rpy': np.array([1.5708, 0.0,    -1.5708]), 'axis': np.array([0, 0, 1])},  # joint_1
    {'xyz': np.array([0.0000,  0.2550, -0.0160]), 'rpy': np.array([0.0,    0.0,     0.0   ]), 'axis': np.array([0, 0, 1])},  # joint_2
    {'xyz': np.array([0.2245,  0.0000,  0.0070]), 'rpy': np.array([-1.5708, 0.0,    0.0   ]), 'axis': np.array([0, 0, 1])},  # joint_3
    {'xyz': np.array([0.0430,  0.0000,  0.0405]), 'rpy': np.array([1.5708, 0.0,     0.0   ]), 'axis': np.array([0, 0, 1])},  # joint_4
    {'xyz': np.array([0.0000,  0.0530,  0.0405]), 'rpy': np.array([-1.5708, 0.0,    0.0   ]), 'axis': np.array([0, 0, 1])},  # joint_5
]

JOINT_NAMES  = [f'joint_{i}' for i in range(6)]
JOINT_LIMITS = [(-3.14159, 3.14159)] * 6   # (lower, upper) rad

# ──────────────────────────────────────────────────────────────────────────────
# WORKSPACE BOUNDS  (Monte Carlo FK, 50 000 samples, + 10 mm safety margin)
# ──────────────────────────────────────────────────────────────────────────────

WORKSPACE_AABB: dict[str, tuple[float, float]] = {
    'x': (-0.576, 0.577),
    'y': (-0.587, 0.593),
    'z': (-0.422, 0.767),
}

MAX_REACH = 0.580   # m – outermost reachable sphere
MIN_REACH = 0.050   # m – inner dead-zone around base


# ──────────────────────────────────────────────────────────────────────────────
# FORWARD KINEMATICS
# ──────────────────────────────────────────────────────────────────────────────

def fk(q: np.ndarray) -> np.ndarray:
    """Return end-effector XYZ (3,) for joint angles q (6,) in radians."""
    T = np.eye(4)
    for i, jd in enumerate(JOINTS):
        R_fixed = Rotation.from_euler('xyz', jd['rpy']).as_matrix()
        T_fixed = np.eye(4)
        T_fixed[:3, :3] = R_fixed
        T_fixed[:3,  3] = jd['xyz']
        R_joint = Rotation.from_rotvec(float(q[i]) * jd['axis']).as_matrix()
        T_joint = np.eye(4)
        T_joint[:3, :3] = R_joint
        T = T @ T_fixed @ T_joint
    return T[:3, 3].copy()


def sample_workspace(n: int = 8_000, seed: int = 0) -> np.ndarray:
    """Sample n random FK positions; returns (n, 3) array."""
    rng = np.random.default_rng(seed)
    lowers = np.array([lo for lo, _ in JOINT_LIMITS])
    uppers = np.array([hi for _, hi in JOINT_LIMITS])
    q_batch = rng.uniform(lowers, uppers, (n, 6))
    return np.array([fk(q) for q in q_batch])


# ──────────────────────────────────────────────────────────────────────────────
# POINT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def check_point(x: float, y: float, z: float) -> tuple[bool, list[str]]:
    violations: list[str] = []

    for axis, val in (('x', x), ('y', y), ('z', z)):
        lo, hi = WORKSPACE_AABB[axis]
        if val < lo:
            violations.append(f'{axis}={val:.4f} m  <  min {lo:.4f} m')
        elif val > hi:
            violations.append(f'{axis}={val:.4f} m  >  max {hi:.4f} m')

    r = float(np.sqrt(x**2 + y**2 + z**2))
    if r > MAX_REACH:
        violations.append(
            f'distance {r:.4f} m  >  max reach {MAX_REACH:.4f} m')
    if r < MIN_REACH:
        violations.append(
            f'distance {r:.4f} m  <  min reach (dead-zone) {MIN_REACH:.4f} m')

    return len(violations) == 0, violations


def clamp_point(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Project (x,y,z) onto the boundary of the safe workspace."""
    x = float(np.clip(x, *WORKSPACE_AABB['x']))
    y = float(np.clip(y, *WORKSPACE_AABB['y']))
    z = float(np.clip(z, *WORKSPACE_AABB['z']))
    r = float(np.sqrt(x**2 + y**2 + z**2))
    if r > MAX_REACH and r > 1e-9:
        s = MAX_REACH / r
        x, y, z = x * s, y * s, z * s
    if r < MIN_REACH and r > 1e-9:
        s = MIN_REACH / r
        x, y, z = x * s, y * s, z * s
    return x, y, z
