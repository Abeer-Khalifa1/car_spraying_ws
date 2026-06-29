#!/usr/bin/env python3
"""
validate_trajectory.py  —  Standalone CLI
==========================================
Validate a Cartesian trajectory CSV against car_spraying_robot's workspace.

Usage
-----
    # report only
    ros2 run trajectory_validator validate_trajectory --ros-args -p csv_path:=/path/to.csv

    # OR run standalone (no ROS required):
    python3 validate_trajectory.py input.csv [--output safe.csv] [--clamp] [--quiet]

Exit codes
----------
  0  all points within workspace
  1  one or more points outside workspace
  2  input error (file not found, bad CSV …)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# When run inside a ROS 2 workspace the package is importable normally.
# When run as a plain script, add the package directory to sys.path.
try:
    from trajectory_validator.robot_workspace import (
        WORKSPACE_AABB, MAX_REACH, MIN_REACH, check_point, clamp_point)
    from trajectory_validator.csv_loader import load_csv, save_csv
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from trajectory_validator.robot_workspace import (
        WORKSPACE_AABB, MAX_REACH, MIN_REACH, check_point, clamp_point)
    from trajectory_validator.csv_loader import load_csv, save_csv


# ──────────────────────────────────────────────────────────────────────────────

def validate(
    input_csv: str,
    output_csv: str | None = None,
    clamp: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Validate *input_csv* and optionally write a corrected *output_csv*.

    Returns
    -------
    dict with keys: total, safe, unsafe, clamped, violations_by_row
    """
    try:
        records = load_csv(input_csv)
    except Exception as exc:
        print(f'ERROR loading CSV: {exc}', file=sys.stderr)
        sys.exit(2)

    W = WORKSPACE_AABB
    print(f'\n{"="*62}')
    print(f'  Trajectory Validator  —  car_spraying_robot (6-DOF)')
    print(f'{"="*62}')
    print(f'  File        : {input_csv}')
    print(f'  Waypoints   : {len(records)}')
    print(f'  X workspace : [{W["x"][0]:+.3f}, {W["x"][1]:+.3f}] m')
    print(f'  Y workspace : [{W["y"][0]:+.3f}, {W["y"][1]:+.3f}] m')
    print(f'  Z workspace : [{W["z"][0]:+.3f}, {W["z"][1]:+.3f}] m')
    print(f'  Max reach   : {MAX_REACH:.3f} m   Min reach: {MIN_REACH:.3f} m')
    print(f'{"="*62}\n')

    safe_count    = 0
    unsafe_count  = 0
    clamped_count = 0
    violations_by_row: dict[int, dict] = {}
    out_records: list[dict] = []

    for rec in records:
        ok, viols = check_point(rec['x'], rec['y'], rec['z'])
        if ok:
            safe_count += 1
            out_records.append(rec)
        else:
            unsafe_count += 1
            violations_by_row[rec['row_index']] = {
                'point': (rec['x'], rec['y'], rec['z']),
                'violations': viols,
            }
            if verbose:
                print(f'  [UNSAFE] Row {rec["row_index"]:4d}  '
                      f'({rec["x"]:+.4f}, {rec["y"]:+.4f}, {rec["z"]:+.4f})')
                for v in viols:
                    print(f'            → {v}')

            if clamp:
                cx, cy, cz = clamp_point(rec['x'], rec['y'], rec['z'])
                clamped_count += 1
                clamped_rec = dict(rec)
                clamped_rec['x'], clamped_rec['y'], clamped_rec['z'] = cx, cy, cz
                out_records.append(clamped_rec)
                if verbose:
                    print(f'           Clamped → ({cx:+.4f}, {cy:+.4f}, {cz:+.4f})')

    print(f'\n{"─"*62}')
    clamp_msg = f'   {clamped_count} clamped' if clamp else ''
    print(f'  RESULTS: {safe_count}/{len(records)} safe   '
          f'{unsafe_count} unsafe{clamp_msg}')
    print(f'{"─"*62}')

    if output_csv is not None:
        if not clamp and unsafe_count > 0:
            print(f'  NOTE: output only written when --clamp is used '
                  f'or all points are safe.')
        else:
            # Recover header line from original file
            raw_lines = Path(input_csv).read_text(encoding='utf-8').splitlines()
            header = None
            for ln in raw_lines:
                if not ln.strip() or ln.strip().startswith('#'):
                    continue
                # Check if it's a header (non-numeric first field)
                try:
                    float(ln.split(',')[0])
                except ValueError:
                    header = ln
                break
            save_csv(out_records, output_csv, header)
            print(f'  Saved   : {output_csv}  ({len(out_records)} rows)')

    return {
        'total': len(records),
        'safe': safe_count,
        'unsafe': unsafe_count,
        'clamped': clamped_count,
        'violations_by_row': violations_by_row,
    }


# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Validate/clamp a Cartesian trajectory CSV '
                    'for car_spraying_robot.')
    parser.add_argument('input',
                        help='Input trajectory CSV')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV (safe/clamped waypoints)')
    parser.add_argument('--clamp', '-c', action='store_true',
                        help='Clamp unsafe points to workspace boundary')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress per-row violation output')
    args = parser.parse_args(argv)

    if not Path(args.input).is_file():
        print(f'ERROR: File not found: {args.input}', file=sys.stderr)
        sys.exit(2)

    result = validate(
        input_csv  = args.input,
        output_csv = args.output,
        clamp      = args.clamp,
        verbose    = not args.quiet,
    )
    sys.exit(0 if result['unsafe'] == 0 else 1)


if __name__ == '__main__':
    main()
