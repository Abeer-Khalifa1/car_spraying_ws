"""
test_trajectory_validator.py
Unit tests — no ROS runtime required.
Run:  pytest test/test_trajectory_validator.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import tempfile
import csv as csv_module

from trajectory_validator.robot_workspace import (
    fk, check_point, clamp_point, WORKSPACE_AABB, MAX_REACH, MIN_REACH)
from trajectory_validator.csv_loader import load_csv, save_csv


# ── FK smoke test ─────────────────────────────────────────────────────────────

def test_fk_home_position():
    """All-zero joint angles → EE should be above the base (z > 0)."""
    pos = fk(np.zeros(6))
    assert pos.shape == (3,)
    assert pos[2] > 0.0, "EE should be above base at home pose"


def test_fk_max_reach_not_exceeded():
    """Any random FK sample should stay within the theoretical max reach sum."""
    rng = np.random.default_rng(42)
    for _ in range(200):
        q = rng.uniform(-3.14, 3.14, 6)
        pos = fk(q)
        r = float(np.linalg.norm(pos))
        assert r < 0.85, f"FK reach {r:.4f} unexpectedly large"


# ── check_point ───────────────────────────────────────────────────────────────

def test_check_point_origin_dead_zone():
    ok, viols = check_point(0.0, 0.0, 0.0)
    assert not ok
    assert any('dead-zone' in v for v in viols)


def test_check_point_safe_centre():
    ok, viols = check_point(0.2, 0.0, 0.3)
    assert ok, f"Expected safe, got: {viols}"


def test_check_point_exceeds_x():
    ok, viols = check_point(0.9, 0.0, 0.3)
    assert not ok
    assert any('x=' in v for v in viols)


def test_check_point_exceeds_reach():
    ok, viols = check_point(0.5, 0.5, 0.5)
    assert not ok
    assert any('distance' in v for v in viols)


def test_check_point_below_z_min():
    ok, viols = check_point(0.1, 0.1, -0.9)
    assert not ok
    assert any('z=' in v for v in viols)


# ── clamp_point ───────────────────────────────────────────────────────────────

def test_clamp_point_far_away():
    cx, cy, cz = clamp_point(5.0, 5.0, 5.0)
    ok, _ = check_point(cx, cy, cz)
    assert ok, "Clamped point should be safe"


def test_clamp_point_already_safe():
    x, y, z = 0.2, 0.1, 0.3
    cx, cy, cz = clamp_point(x, y, z)
    assert abs(cx - x) < 1e-9 and abs(cy - y) < 1e-9 and abs(cz - z) < 1e-9


def test_clamp_point_exceeds_x():
    cx, cy, cz = clamp_point(0.9, 0.0, 0.3)
    assert cx <= WORKSPACE_AABB['x'][1] + 1e-9


# ── CSV loader ────────────────────────────────────────────────────────────────

def _write_tmp_csv(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                    delete=False, encoding='utf-8')
    f.write(content)
    f.close()
    return f.name


def test_load_csv_with_header():
    path = _write_tmp_csv("x,y,z\n0.1,0.2,0.3\n0.4,0.5,0.6\n")
    recs = load_csv(path)
    assert len(recs) == 2
    assert recs[0]['x'] == pytest.approx(0.1)
    assert recs[1]['z'] == pytest.approx(0.6)
    os.unlink(path)


def test_load_csv_with_time_and_orient():
    path = _write_tmp_csv(
        "time,x,y,z,roll,pitch,yaw\n"
        "0.0,0.1,0.2,0.3,0.0,0.0,0.0\n"
        "0.5,0.2,0.3,0.4,0.1,0.1,0.1\n"
    )
    recs = load_csv(path)
    assert len(recs) == 2
    assert recs[0]['time'] == pytest.approx(0.0)
    assert recs[1]['roll'] == pytest.approx(0.1)
    os.unlink(path)


def test_load_csv_no_header():
    path = _write_tmp_csv("0.1,0.2,0.3\n0.4,0.5,0.6\n")
    recs = load_csv(path)
    assert len(recs) == 2
    assert recs[0]['x'] == pytest.approx(0.1)
    os.unlink(path)


def test_load_csv_skips_comments():
    path = _write_tmp_csv(
        "# comment line\n"
        "x,y,z\n"
        "# another comment\n"
        "0.1,0.2,0.3\n"
    )
    recs = load_csv(path)
    assert len(recs) == 1
    os.unlink(path)


def test_save_and_reload_csv():
    path_in  = _write_tmp_csv("x,y,z\n0.1,0.2,0.3\n0.4,0.5,0.6\n")
    path_out = path_in + '_out.csv'
    recs = load_csv(path_in)
    # Mutate one record
    recs[0]['x'] = 0.15
    save_csv(recs, path_out, header='x,y,z')
    recs2 = load_csv(path_out)
    assert recs2[0]['x'] == pytest.approx(0.15)
    os.unlink(path_in)
    os.unlink(path_out)
