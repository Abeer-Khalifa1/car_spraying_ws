#!/usr/bin/env python3
"""
filter_and_forward.py
=====================
Validates peya.csv against the robot workspace, strips unreachable waypoints
(if they are ≤ 10 % of the total), writes a clean peya_validated.csv, then
publishes the validated CSV path on the ROS 2 topic
  /trajectory_validator/validated_csv_path   (std_msgs/String, TRANSIENT_LOCAL)

square_xz.cpp reads its CSV path from the ROS 2 parameter  csv_path  which can
be overridden at launch time — so no recompile is needed to point it at the
validated file.

Exit codes
----------
  0  validation passed (with or without stripped points)
  1  too many unreachable points (> 10 %) — path rejected, square_xz NOT started
  2  input error (file not found, bad CSV …)

Usage (standalone — no ROS required)
-------------------------------------
    python3 filter_and_forward.py /path/to/peya.csv \\
            --output /path/to/peya_validated.csv \\
            [--threshold 0.10]

Usage as a ROS 2 node (launched before square_xz)
---------------------------------------------------
    ros2 run trajectory_validator filter_and_forward \\
        --ros-args \\
        -p csv_path:=/home/user/car_spraying_ws/src/square_trajectory/peya.csv \\
        -p output_path:=/home/user/car_spraying_ws/src/square_trajectory/peya_validated.csv \\
        -p threshold:=0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── allow both standalone and ROS-installed import ───────────────────────────
try:
    from trajectory_validator.robot_workspace import check_point
    from trajectory_validator.csv_loader import load_csv, save_csv
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from trajectory_validator.robot_workspace import check_point
    from trajectory_validator.csv_loader import load_csv, save_csv


# ─────────────────────────────────────────────────────────────────────────────
# Core logic (ROS-free — usable in unit tests too)
# ─────────────────────────────────────────────────────────────────────────────

def filter_trajectory(
    input_csv: str,
    output_csv: str,
    threshold: float = 0.10,
    verbose: bool = True,
) -> dict:
    """
    Validate *input_csv*, strip unreachable points if within *threshold*, and
    write the result to *output_csv*.

    Returns
    -------
    dict with keys:
        total       – total waypoints in input
        safe        – waypoints within workspace
        unsafe      – waypoints outside workspace
        unsafe_pct  – fraction of total that are unsafe (0.0 – 1.0)
        passed      – True if unsafe_pct <= threshold
        output_path – path written (None if rejected)
    """
    try:
        records = load_csv(input_csv)
    except Exception as exc:
        print(f'[filter_and_forward] ERROR loading CSV: {exc}', file=sys.stderr)
        sys.exit(2)

    if not records:
        print('[filter_and_forward] ERROR: CSV contains no valid rows.', file=sys.stderr)
        sys.exit(2)

    total = len(records)
    safe_records   = []
    unsafe_records = []

    for rec in records:
        ok, violations = check_point(rec['x'], rec['y'], rec['z'])
        if ok:
            safe_records.append(rec)
        else:
            unsafe_records.append((rec, violations))

    n_safe   = len(safe_records)
    n_unsafe = len(unsafe_records)
    unsafe_pct = n_unsafe / total

    # ── summary banner ───────────────────────────────────────────────────────
    if verbose:
        sep = '=' * 62
        print(f'\n{sep}')
        print(f'  filter_and_forward  —  car_spraying_robot')
        print(f'{sep}')
        print(f'  Input CSV   : {input_csv}')
        print(f'  Total pts   : {total}')
        print(f'  Safe        : {n_safe}  ({100*(n_safe/total):.1f} %)')
        print(f'  Unsafe      : {n_unsafe}  ({100*unsafe_pct:.1f} %)')
        print(f'  Threshold   : ≤ {100*threshold:.0f} % unsafe allowed')
        print(f'{sep}')

        if unsafe_records:
            print()
            for rec, viols in unsafe_records:
                print(f'  [UNSAFE] Row {rec["row_index"]:4d}  '
                      f'({rec["x"]:+.4f}, {rec["y"]:+.4f}, {rec["z"]:+.4f})')
                for v in viols:
                    print(f'           → {v}')
            print()

    # ── decision ─────────────────────────────────────────────────────────────
    passed = unsafe_pct <= threshold

    if not passed:
        msg = (
            f'\n[filter_and_forward] ✗ REJECTED — {n_unsafe}/{total} waypoints '
            f'({100*unsafe_pct:.1f} %) are outside the workspace, '
            f'which exceeds the {100*threshold:.0f} % threshold.\n'
            f'  The trajectory cannot be safely executed. '
            f'Check your path generator or workspace limits.\n'
            f'  square_xz will NOT be started.'
        )
        print(msg, file=sys.stderr)
        return {
            'total': total,
            'safe': n_safe,
            'unsafe': n_unsafe,
            'unsafe_pct': unsafe_pct,
            'passed': False,
            'output_path': None,
        }

    # ── write filtered CSV ───────────────────────────────────────────────────
    # Recover original header line (if any) to preserve column names
    raw_lines = Path(input_csv).read_text(encoding='utf-8').splitlines()
    header = None
    for ln in raw_lines:
        stripped = ln.strip()
        if not stripped or stripped.startswith('#'):
            continue
        try:
            float(stripped.split(',')[0])
        except ValueError:
            header = ln  # it's a text header row
        break

    save_csv(safe_records, output_csv, header)

    if verbose:
        status = 'PASSED (all points safe)' if n_unsafe == 0 else \
                 f'PASSED — {n_unsafe} unsafe point(s) stripped'
        print(f'  ✓ {status}')
        print(f'  Output CSV  : {output_csv}  ({n_safe} rows)')
        print(f'  Ready for square_xz.\n')

    return {
        'total': total,
        'safe': n_safe,
        'unsafe': n_unsafe,
        'unsafe_pct': unsafe_pct,
        'passed': True,
        'output_path': output_csv,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 node (optional — publishes validated path on a latched topic)
# ─────────────────────────────────────────────────────────────────────────────

def ros_main(args=None) -> None:
    """
    ROS 2 entry point.  Reads parameters, runs filter_trajectory(), and
    publishes the output path on /trajectory_validator/validated_csv_path
    with TRANSIENT_LOCAL durability so late subscribers (square_xz) can
    receive it even if they start after this node finishes.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init(args=args)

    node = rclpy.create_node('filter_and_forward')

    node.declare_parameter('csv_path', '')
    node.declare_parameter('output_path', '')
    node.declare_parameter('threshold', 0.10)

    csv_path    = node.get_parameter('csv_path').value
    output_path = node.get_parameter('output_path').value
    threshold   = float(node.get_parameter('threshold').value)

    if not csv_path:
        node.get_logger().error(
            'Parameter csv_path is empty. '
            'Pass: --ros-args -p csv_path:=/path/to/peya.csv')
        rclpy.shutdown()
        sys.exit(2)

    if not output_path:
        # default: peya_validated.csv next to the input
        p = Path(csv_path)
        output_path = str(p.parent / (p.stem + '_validated' + p.suffix))

    # Run the filter (verbose to logger-friendly stdout)
    result = filter_trajectory(csv_path, output_path, threshold=threshold)

    if not result['passed']:
        node.get_logger().error(
            f'Trajectory REJECTED: {result["unsafe"]}/{result["total"]} '
            f'waypoints ({100*result["unsafe_pct"]:.1f} %) out of workspace '
            f'(threshold {100*threshold:.0f} %). square_xz will NOT start.')
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info(
        f'Trajectory validated: {result["safe"]}/{result["total"]} safe '
        f'({result["unsafe"]} stripped). '
        f'Output: {output_path}')

    # Publish validated path with TRANSIENT_LOCAL so square_xz can receive it
    # even if it starts a few seconds later.
    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    pub = node.create_publisher(String, '/trajectory_validator/validated_csv_path', qos)
    msg = String()
    msg.data = output_path
    pub.publish(msg)

    node.get_logger().info(
        f'Published validated CSV path on /trajectory_validator/validated_csv_path')

    # Spin briefly so TRANSIENT_LOCAL message is delivered to any late subscribers
    import time
    deadline = time.time() + 2.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.try_shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    """
    Dual-mode entry point:
      • If --ros-args is present → ROS 2 node mode
      • Otherwise               → standalone CLI mode
    """
    raw = argv if argv is not None else sys.argv[1:]

    if '--ros-args' in raw:
        ros_main()
        return

    parser = argparse.ArgumentParser(
        description='Validate & filter a Cartesian trajectory CSV for car_spraying_robot.')
    parser.add_argument('input',
                        help='Input trajectory CSV (e.g. peya.csv)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path (default: <input>_validated.csv)')
    parser.add_argument('--threshold', '-t', type=float, default=0.10,
                        help='Max fraction of unsafe points allowed (default 0.10 = 10 %%)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress per-row violation output')
    args = parser.parse_args(raw)

    if not Path(args.input).is_file():
        print(f'ERROR: File not found: {args.input}', file=sys.stderr)
        sys.exit(2)

    output = args.output
    if output is None:
        p = Path(args.input)
        output = str(p.parent / (p.stem + '_validated' + p.suffix))

    result = filter_trajectory(
        input_csv  = args.input,
        output_csv = output,
        threshold  = args.threshold,
        verbose    = not args.quiet,
    )

    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()