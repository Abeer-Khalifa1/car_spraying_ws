"""
csv_loader.py
=============
Flexible trajectory CSV reader.

Accepted column layouts (case-insensitive, auto-detected):
    x, y, z
    x, y, z, roll, pitch, yaw
    time, x, y, z
    time, x, y, z, roll, pitch, yaw
    (no header — positional: col 0=x, 1=y, 2=z)

Lines starting with '#' are treated as comments and skipped.
"""

from __future__ import annotations
import csv
import io
from pathlib import Path
from typing import Any


# ── helpers ───────────────────────────────────────────────────────────────────

_ALIASES: dict[str, list[str]] = {
    'x':     ['x', 'pos_x', 'position_x'],
    'y':     ['y', 'pos_y', 'position_y'],
    'z':     ['z', 'pos_z', 'position_z'],
    'roll':  ['roll', 'rx', 'r'],
    'pitch': ['pitch', 'ry', 'p'],
    'yaw':   ['yaw', 'rz', 'yw'],
    'time':  ['time', 't', 'timestamp', 'sec'],
}

def _map_header(raw_header: list[str]) -> dict[str, int]:
    h = [c.strip().lower() for c in raw_header]
    mapping: dict[str, int] = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in h:
                mapping[canonical] = h.index(alias)
                break
    return mapping


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a trajectory CSV.

    Returns a list of records, each dict containing at minimum:
        row_index (int), x (float), y (float), z (float), raw_row (list[str])
    Optionally also: time, roll, pitch, yaw.
    """
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith('#')]
    if not lines:
        raise ValueError(f'CSV is empty or only comments: {path}')

    all_rows = list(csv.reader(lines))

    # Detect header vs. pure-data first row
    try:
        float(all_rows[0][0])
        float(all_rows[0][1])
        has_header = False
        data_rows  = all_rows
        mapping: dict[str, int] = {'x': 0, 'y': 1, 'z': 2}
    except (ValueError, IndexError):
        has_header = True
        mapping    = _map_header(all_rows[0])
        data_rows  = all_rows[1:]

    required = {'x', 'y', 'z'}
    if not required.issubset(mapping):
        raise ValueError(
            f'Cannot find x,y,z columns in {path}. '
            f'Detected mapping: {mapping}. '
            f'Header: {all_rows[0] if has_header else "(no header)"}')

    records: list[dict[str, Any]] = []
    for i, row in enumerate(data_rows):
        if not any(c.strip() for c in row):
            continue
        try:
            rec: dict[str, Any] = {
                'row_index': i + (2 if has_header else 1),
                'x': float(row[mapping['x']]),
                'y': float(row[mapping['y']]),
                'z': float(row[mapping['z']]),
                'raw_row': list(row),
                '_col_x': mapping['x'],
                '_col_y': mapping['y'],
                '_col_z': mapping['z'],
            }
            for key in ('time', 'roll', 'pitch', 'yaw'):
                if key in mapping:
                    try:
                        rec[key] = float(row[mapping[key]])
                    except (IndexError, ValueError):
                        pass
            records.append(rec)
        except (IndexError, ValueError) as exc:
            import warnings
            warnings.warn(f'Skipping row {i}: {exc}')

    return records


def save_csv(
    records: list[dict[str, Any]],
    path: str | Path,
    header: str | None = None,
) -> None:
    """Write (possibly clamped) records back to a CSV file."""
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        if header:
            fh.write(header.rstrip('\n') + '\n')
        writer = csv.writer(fh)
        for rec in records:
            row = list(rec['raw_row'])
            row[rec['_col_x']] = f"{rec['x']:.6f}"
            row[rec['_col_y']] = f"{rec['y']:.6f}"
            row[rec['_col_z']] = f"{rec['z']:.6f}"
            writer.writerow(row)
