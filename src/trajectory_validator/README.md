# trajectory_validator (ROS 2 Jazzy)

Validates and visualises Cartesian trajectory CSVs against the reachable
workspace of **car_spraying_robot** (UR3-like 6-DOF arm, SolidWorks → URDF).

---

## Workspace bounds (auto-derived from URDF via 50 000-sample Monte Carlo FK)

| Axis | Min (m) | Max (m) |
|------|---------|---------|
| X    | −0.576  | +0.577  |
| Y    | −0.587  | +0.593  |
| Z    | −0.422  | +0.767  |
| Reach radius | 0.050 (dead-zone) | 0.580 |

---

## Package layout

```
trajectory_validator/
├── trajectory_validator/
│   ├── __init__.py
│   ├── robot_workspace.py          ← FK, workspace bounds, check/clamp helpers
│   ├── csv_loader.py               ← flexible CSV reader/writer (ROS-free)
│   ├── trajectory_validator_node.py← ROS 2 node
│   ├── validate_trajectory.py      ← standalone CLI + importable library
│   └── visualize_workspace.py      ← matplotlib 3-D viewer
├── launch/
│   └── validate.launch.py
├── config/
│   ├── workspace.yaml
│   ├── rviz2.rviz
│   └── sample_trajectory.csv
├── test/
│   └── test_trajectory_validator.py
├── resource/trajectory_validator   ← ament marker
├── package.xml
├── setup.py
└── setup.cfg
```

---

## 1 — Build & install

```bash
cd ~/ros2_ws/src
cp -r /path/to/trajectory_validator .
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select trajectory_validator
source install/setup.bash
```

---

## 2 — ROS 2 node

### Launch

```bash
# Basic
ros2 launch trajectory_validator validate.launch.py \
    csv_path:=$(pwd)/config/sample_trajectory.csv

# With RViz2
ros2 launch trajectory_validator validate.launch.py \
    csv_path:=/abs/path/to/traj.csv \
    rviz:=true

# Auto-clamp log + custom frame
ros2 launch trajectory_validator validate.launch.py \
    csv_path:=/abs/path/to/traj.csv \
    frame_id:=world  clamp:=true  rate_hz:=2.0
```

### Parameters (settable at any time)

| Parameter | Type   | Default      | Description                        |
|-----------|--------|--------------|------------------------------------|
| csv_path  | string | ''           | Absolute path to trajectory CSV    |
| frame_id  | string | 'base_link'  | TF frame for RViz2 markers         |
| rate_hz   | double | 1.0          | Marker republish rate (Hz)         |
| clamp     | bool   | false        | Log clamped position alongside violation |

### Reload CSV at runtime (no restart needed)

```bash
ros2 param set /trajectory_validator csv_path /new/path/trajectory.csv
ros2 service call /trajectory_validator/reload std_srvs/srv/Trigger {}
```

### Published topics

| Topic | Message type | Content |
|-------|-------------|---------|
| `~/trajectory_markers` | `visualization_msgs/MarkerArray` | 🟢 safe spheres · 🔴 unsafe X · path line · AABB box |
| `~/workspace_sphere`   | `visualization_msgs/Marker`      | Translucent max-reach sphere |

---

## 3 — Standalone CLI (no ROS)

```bash
# Report only (exit 0 = all safe, exit 1 = some unsafe)
python3 trajectory_validator/validate_trajectory.py config/sample_trajectory.csv

# Clamp unsafe points → write corrected CSV
python3 trajectory_validator/validate_trajectory.py config/sample_trajectory.csv \
    --output safe_traj.csv --clamp

# Quiet mode (only summary)
python3 trajectory_validator/validate_trajectory.py traj.csv --quiet
```

### Or via ros2 run

```bash
ros2 run trajectory_validator validate_trajectory \
    config/sample_trajectory.csv --clamp --output safe.csv
```

---

## 4 — Visualiser (matplotlib, no ROS)

```bash
# Workspace cloud only
ros2 run trajectory_validator visualize_workspace

# With trajectory overlay (green=safe, red X=unsafe)
ros2 run trajectory_validator visualize_workspace \
    config/sample_trajectory.csv

# Save PNG
ros2 run trajectory_validator visualize_workspace \
    config/sample_trajectory.csv workspace.png
```

---

## 5 — Python library

```python
from trajectory_validator import check_point, clamp_point, load_csv

# Single point
ok, violations = check_point(x=0.3, y=0.1, z=0.4)
if not ok:
    for v in violations:
        print(v)

# Clamp to boundary
cx, cy, cz = clamp_point(0.9, 0.9, 0.9)

# Validate a whole file
from trajectory_validator.validate_trajectory import validate
result = validate('my_traj.csv', output_csv='safe.csv', clamp=True)
print(result['unsafe'], "unsafe points")
```

---

## 6 — CSV format

Auto-detected (case-insensitive, any column order):

```
x, y, z
x, y, z, roll, pitch, yaw
time, x, y, z
time, x, y, z, roll, pitch, yaw
```

Lines starting with `#` are comments and ignored.  
No-header CSVs also work (positional: col 0 = x, col 1 = y, col 2 = z).

---

## 7 — Tests

```bash
cd ~/ros2_ws
colcon test --packages-select trajectory_validator
colcon test-result --verbose

# Or run pytest directly (no ROS needed)
pytest src/trajectory_validator/test/ -v
```

---

## Differences from ROS 1 version

| Feature | ROS 1 | ROS 2 Jazzy |
|---------|-------|-------------|
| Build system | catkin | ament_python |
| Launch format | XML `.launch` | Python `.launch.py` |
| Parameter API | `rospy.get_param` | `Node.declare_parameter` + `get_parameter` |
| Param hot-reload | `/reload` service only | `ros2 param set` + `/reload` service |
| Node namespace | `/trajectory_markers` | `~/trajectory_markers` (private) |
| Marker latching | `latch=True` | `Durability: Transient Local` (set in RViz config) |
| Service type | `std_srvs/Trigger` | `std_srvs/srv/Trigger` |
