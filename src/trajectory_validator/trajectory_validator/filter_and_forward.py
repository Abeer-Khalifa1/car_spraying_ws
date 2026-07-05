#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── allow both standalone and ROS-installed import ───────────────────────────
try:
    from trajectory_validator.robot_workspace import check_point, clamp_point
    from trajectory_validator.csv_loader import load_csv, save_csv
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from trajectory_validator.robot_workspace import check_point, clamp_point
    from trajectory_validator.csv_loader import load_csv, save_csv


# ─────────────────────────────────────────────────────────────────────────────
# Core logic (ROS-free — usable in unit tests too)
# ─────────────────────────────────────────────────────────────────────────────

def filter_trajectory(
    input_csv: str,
    output_csv: str,
    threshold: float = 0.10,
    verbose: bool = True,
    standoff: float = 0.20,
    clamp: bool = True,
    max_clamp_correction: float = 0.002,
) -> dict:

    try:
        records = load_csv(input_csv)
    except Exception as exc:
        print(f'[filter_and_forward] ERROR loading CSV: {exc}', file=sys.stderr)
        sys.exit(2)

    if not records:
        print('[filter_and_forward] ERROR: CSV contains no valid rows.', file=sys.stderr)
        sys.exit(2)

    # Check whether normals were loaded
    has_normals = all('nx' in rec and 'ny' in rec and 'nz' in rec for rec in records)
    if not has_normals:
        print(
            '[filter_and_forward] WARNING: Surface normal columns (7-9 / nx,ny,nz) '
            'not found in CSV. Falling back to validating raw surface points. '
            'This may pass points whose nozzle positions are outside the workspace.',
            file=sys.stderr,
        )

    total = len(records)
    safe_records    = []
    clamped_records = []
    unsafe_records  = []

    for rec in records:
        if has_normals:
            # Compute actual nozzle position — this is what the arm moves to.
            # Matches square_xz.cpp pose_from_surface():
            #   pose.position.x = wp.x - standoff * wp.nx  (and same for y, z)
            import math
            nx, ny, nz = rec['nx'], rec['ny'], rec['nz']
            n_len = math.sqrt(nx*nx + ny*ny + nz*nz)
            if n_len > 1e-9:
                nx, ny, nz = nx/n_len, ny/n_len, nz/n_len
            check_x = rec['x'] - standoff * nx
            check_y = rec['y'] - standoff * ny
            check_z = rec['z'] - standoff * nz
        else:
            check_x, check_y, check_z = rec['x'], rec['y'], rec['z']

        ok, violations = check_point(check_x, check_y, check_z)
        if ok:
            safe_records.append(rec)
            continue

        if clamp:
            import math
            ccx, ccy, ccz = clamp_point(check_x, check_y, check_z)
            correction = math.sqrt(
                (ccx - check_x) ** 2 + (ccy - check_y) ** 2 + (ccz - check_z) ** 2)

            if correction <= max_clamp_correction:
                # Nudge the surface point by the same delta as the nozzle
                # correction so the clamped nozzle position lands exactly
                # on the workspace boundary.
                dx, dy, dz = ccx - check_x, ccy - check_y, ccz - check_z
                clamped_rec = dict(rec)
                clamped_rec['x'] = rec['x'] + dx
                clamped_rec['y'] = rec['y'] + dy
                clamped_rec['z'] = rec['z'] + dz
                safe_records.append(clamped_rec)
                clamped_records.append((rec, violations, correction))
                continue
            # else: correction too large — falls through to unsafe/dropped

        unsafe_records.append((rec, violations, check_x, check_y, check_z))

    n_safe    = len(safe_records) - len(clamped_records)
    n_clamped = len(clamped_records)
    n_unsafe  = len(unsafe_records)
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
        if clamp:
            print(f'  Clamped     : {n_clamped}  ({100*(n_clamped/total):.1f} %)  '
                  f'[correction <= {1000*max_clamp_correction:.1f} mm]')
        print(f'  Unsafe      : {n_unsafe}  ({100*unsafe_pct:.1f} %)  (dropped)')
        print(f'  Threshold   : ≤ {100*threshold:.0f} % unsafe allowed')
        print(f'{sep}')

        if clamped_records:
            print()
            for rec, viols, correction in clamped_records:
                print(f'  [CLAMPED] Row {rec["row_index"]:4d}  '
                      f'surface=({rec["x"]:+.4f}, {rec["y"]:+.4f}, {rec["z"]:+.4f})  '
                      f'correction={1000*correction:.2f} mm')
                for v in viols:
                    print(f'           → {v}')
            print()

        if unsafe_records:
            print()
            lbl = 'NOZZLE' if has_normals else 'SURFACE'
            for rec, viols, cx, cy, cz in unsafe_records:
                print(f'  [UNSAFE] Row {rec["row_index"]:4d}  '
                      f'surface=({rec["x"]:+.4f}, {rec["y"]:+.4f}, {rec["z"]:+.4f})  '
                      f'{lbl}=({cx:+.4f}, {cy:+.4f}, {cz:+.4f})')
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
            'clamped': n_clamped,
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
        bits = []
        if n_clamped:
            bits.append(f'{n_clamped} clamped')
        if n_unsafe:
            bits.append(f'{n_unsafe} unsafe point(s) stripped')
        status = 'PASSED (all points safe)' if not bits else f'PASSED — {", ".join(bits)}'
        print(f'  ✓ {status}')
        print(f'  Output CSV  : {output_csv}  ({len(safe_records)} rows)')
        print(f'  Ready for square_xz.\n')

    return {
        'total': total,
        'safe': n_safe,
        'clamped': n_clamped,
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
    node.declare_parameter('clamp', True)
    node.declare_parameter('max_clamp_correction', 0.002)

    csv_path    = node.get_parameter('csv_path').value
    output_path = node.get_parameter('output_path').value
    threshold   = float(node.get_parameter('threshold').value)
    clamp       = bool(node.get_parameter('clamp').value)
    max_clamp_correction = float(node.get_parameter('max_clamp_correction').value)

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
    result = filter_trajectory(csv_path, output_path, threshold=threshold,
                                clamp=clamp, max_clamp_correction=max_clamp_correction)

    if not result['passed']:
        node.get_logger().error(
            f'Trajectory REJECTED: {result["unsafe"]}/{result["total"]} '
            f'waypoints ({100*result["unsafe_pct"]:.1f} %) out of workspace '
            f'(threshold {100*threshold:.0f} %). square_xz will NOT start.')
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info(
        f'Trajectory validated: {result["safe"]}/{result["total"]} safe, '
        f'{result["clamped"]} clamped, {result["unsafe"]} stripped. '
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
    parser.add_argument('--no-clamp', dest='clamp', action='store_false',
                        help='Disable clamping — strip all unsafe points instead (old behavior)')
    parser.add_argument('--max-clamp-correction', type=float, default=0.002,
                        help='Max correction distance in metres to allow when clamping '
                             '(default 0.002 = 2 mm); larger corrections are dropped instead')
    parser.set_defaults(clamp=True)
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
        clamp      = args.clamp,
        max_clamp_correction = args.max_clamp_correction,
    )

    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()