#!/usr/bin/env python3

from __future__ import annotations
import sys
import os
import numpy as np

try:
    from trajectory_validator.robot_workspace import (
        WORKSPACE_AABB, MAX_REACH, MIN_REACH,
        sample_workspace, check_point,
    )
    from trajectory_validator.csv_loader import load_csv
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from trajectory_validator.robot_workspace import (
        WORKSPACE_AABB, MAX_REACH, MIN_REACH,
        sample_workspace, check_point,
    )
    from trajectory_validator.csv_loader import load_csv

import matplotlib
if not os.environ.get('DISPLAY') and sys.platform != 'darwin':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _draw_aabb_3d(ax) -> None:
    xl, xh = WORKSPACE_AABB['x']
    yl, yh = WORKSPACE_AABB['y']
    zl, zh = WORKSPACE_AABB['z']
    corners = np.array([
        [xl,yl,zl],[xh,yl,zl],[xh,yh,zl],[xl,yh,zl],
        [xl,yl,zh],[xh,yl,zh],[xh,yh,zh],[xl,yh,zh],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot3D(*zip(corners[a], corners[b]),
                  color='#FF8C00', lw=0.9, alpha=0.55)


def visualize(trajectory_csv: str | None = None, save_png: str | None = None) -> None:
    print('Sampling workspace (FK cloud)…', end=' ', flush=True)
    pts = sample_workspace(n=8_000)
    print(f'done  ({len(pts)} pts)')

    # ── load trajectory if given ──────────────────────────────────────────
    traj_safe: list[list[float]] = []
    traj_bad:  list[list[float]] = []
    if trajectory_csv and os.path.isfile(trajectory_csv):
        for rec in load_csv(trajectory_csv):
            ok, _ = check_point(rec['x'], rec['y'], rec['z'])
            (traj_safe if ok else traj_bad).append(
                [rec['x'], rec['y'], rec['z']])
        print(f'Trajectory: {len(traj_safe)} safe, {len(traj_bad)} unsafe')

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('car_spraying_robot — Reachable Workspace',
                 fontsize=14, fontweight='bold', y=0.99)

    # ── 3-D scatter ───────────────────────────────────────────────────────
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax3d.scatter(pts[:,0], pts[:,1], pts[:,2],
                      c=pts[:,2], cmap='viridis', s=1, alpha=0.12)
    _draw_aabb_3d(ax3d)
    ax3d.scatter([0],[0],[0], c='red', s=70, zorder=5, label='Base origin')

    if traj_safe:
        s = np.array(traj_safe)
        ax3d.plot(s[:,0], s[:,1], s[:,2], 'g-o', ms=4, lw=1.5,
                  label=f'Safe ({len(traj_safe)})', zorder=6)
    if traj_bad:
        b = np.array(traj_bad)
        ax3d.scatter(b[:,0], b[:,1], b[:,2], c='red', s=50,
                     marker='X', zorder=7, label=f'UNSAFE ({len(traj_bad)})')

    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3-D View')
    ax3d.legend(fontsize=8, loc='upper left')
    fig.colorbar(sc, ax=ax3d, shrink=0.5, label='Z height (m)')

    # ── top-down (XY) ─────────────────────────────────────────────────────
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.scatter(pts[:,0], pts[:,1], c=pts[:,2], cmap='viridis',
                  s=0.5, alpha=0.18)
    xl, xh = WORKSPACE_AABB['x']; yl, yh = WORKSPACE_AABB['y']
    ax_xy.add_patch(Rectangle((xl, yl), xh-xl, yh-yl,
                                fill=False, edgecolor='#FF8C00', lw=1.4))
    ax_xy.add_patch(Circle((0,0), MAX_REACH,
                            fill=False, edgecolor='crimson', lw=1.2,
                            linestyle='--', label=f'Max reach {MAX_REACH} m'))
    ax_xy.add_patch(Circle((0,0), MIN_REACH,
                            fill=False, edgecolor='dimgray', lw=1.0,
                            linestyle=':', label=f'Dead-zone {MIN_REACH} m'))
    if traj_safe:
        s = np.array(traj_safe)
        ax_xy.plot(s[:,0], s[:,1], 'g-o', ms=3, lw=1.2)
    if traj_bad:
        b = np.array(traj_bad)
        ax_xy.scatter(b[:,0], b[:,1], c='red', s=45, marker='X', zorder=5)
    ax_xy.scatter([0],[0], c='red', s=60, zorder=6)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top-Down View (XY plane)')
    ax_xy.legend(fontsize=7); ax_xy.grid(True, alpha=0.25)

    # ── side view (XZ) ────────────────────────────────────────────────────
    ax_xz = fig.add_subplot(2, 2, 4)
    ax_xz.scatter(pts[:,0], pts[:,2], c=pts[:,1], cmap='plasma',
                  s=0.5, alpha=0.18)
    zl, zh = WORKSPACE_AABB['z']
    ax_xz.add_patch(Rectangle((xl, zl), xh-xl, zh-zl,
                                fill=False, edgecolor='#FF8C00', lw=1.4))
    if traj_safe:
        s = np.array(traj_safe)
        ax_xz.plot(s[:,0], s[:,2], 'g-o', ms=3, lw=1.2)
    if traj_bad:
        b = np.array(traj_bad)
        ax_xz.scatter(b[:,0], b[:,2], c='red', s=45, marker='X', zorder=5)
    ax_xz.scatter([0],[0], c='red', s=60, zorder=6)
    ax_xz.set_aspect('equal')
    ax_xz.set_xlabel('X (m)'); ax_xz.set_ylabel('Z (m)')
    ax_xz.set_title('Side View (XZ plane)')
    ax_xz.grid(True, alpha=0.25)

    plt.tight_layout()

    if save_png:
        plt.savefig(save_png, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_png}')
    else:
        plt.show()


def main(argv=None) -> None:
    args = (argv or sys.argv)[1:]
    csv_path = args[0] if len(args) >= 1 else None
    png_path = args[1] if len(args) >= 2 else None
    visualize(csv_path, png_path)


if __name__ == '__main__':
    main()
