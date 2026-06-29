#!/usr/bin/env python3

"""
coverage_quality_node.py  (improved visualisation)
========================================

Real-time coating quality estimation.

Nozzle: Lmuwnm ST-6 Automatic Spray Gun, ø1.0mm orifice
Coverage unit mapping (simulation → physical):
    coverage value = accumulated voxel hit count from spray_sim_node
    ST-6 ø1.0mm SMD ≈ 15 µm per deposit layer
    Single boustrophedon pass ≈ 1–4 hits per surface voxel at 10 Hz.
    ----------------------------------------------------------------
    Automotive OEM basecoat target: 13–38 µm  (per MDPI Coatings 2016)
    Corresponding coverage thresholds (calibrated to sim hit counts):
        < 1   : unpainted  — zero hits, no film formed
        1     : weak       — below min adhesion (~13 µm equivalent)
        2–6   : good       — target OEM basecoat window (13–38 µm)
        > 6   : overspray  — excess material, risk of runs/sags

Subscribes:
    /spray/coverage_cloud   (PointCloud2: x, y, z, intensity=coverage)

Publishes:
    /spray/reward           (Float32)
    /spray/unpainted_cloud  (PointCloud2)
    /spray/good_cloud       (PointCloud2)
    /spray/overspray_cloud  (PointCloud2)

Visualisation layout (2×2 + status bar):
    Top-left    — 3D scatter coloured by CONTINUOUS coverage heatmap
    Top-right   — 2D Y-Z projection with heatmap + contour boundary
    Bottom-left — Coverage histogram (distribution of hit counts)
    Bottom-right— Zone breakdown bar chart
    Top strip   — Large status panel (Good%, Coverage%, Reward) with traffic-light colour
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import threading
import struct

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ── thresholds ──────────────────────────────────────────────────
# Calibrated to spray_sim_node output: 10 Hz timer, amount=1.0/voxel/tick,
# voxel dedup per tick → a single boustrophedon pass produces ~1–4 hits.
# Previous values (2/5/12) required 5+ hits for "good", unreachable in 1 pass.
THR_UNPAINTED = 1.0   # < 1   → unpainted  (zero hits)
THR_GOOD_LO   = 2.0   # 2–6   → good OEM window  (1–2 full passes)
THR_GOOD_HI   = 6.0   # > 6   → overspray  (3+ passes, diminishing returns)

# ── zone colours (kept for bar chart) ──────────────────────────
C_UNPAINTED = '#1a6faf'   # steel blue
C_WEAK      = '#f7d04b'   # amber
C_GOOD      = '#2dbe4e'   # lime green
C_OVER      = '#e03030'   # red

# ── continuous colourmap for heatmap views ──────────────────────
# 'RdYlGn': red (no paint) → yellow (weak) → green (good) → ...
# We cap the upper end so overspray doesn't look the same as good.
CMAP_NAME  = 'RdYlGn'
CMAP       = cm.get_cmap(CMAP_NAME)
NORM_VMIN  = 0.0
NORM_VMAX  = THR_GOOD_HI   # clip display at overspray threshold


def _coverage_to_colour(cov: np.ndarray) -> np.ndarray:
    """Map raw coverage counts → RGBA using RdYlGn, clipped at NORM_VMAX."""
    norm = mcolors.Normalize(vmin=NORM_VMIN, vmax=NORM_VMAX, clip=True)
    return CMAP(norm(cov))


class CoverageQualityNode(Node):

    def __init__(self):
        super().__init__('coverage_quality_node')

        # ── parameters ──────────────────────────────────────────
        self.declare_parameter('min_thickness', THR_GOOD_LO)
        self.declare_parameter('max_thickness', THR_GOOD_HI)
        self.MIN = self.get_parameter('min_thickness').value
        self.MAX = self.get_parameter('max_thickness').value

        # ── state ───────────────────────────────────────────────
        self.pts      = np.zeros((0, 4), dtype=np.float32)
        self.lock     = threading.Lock()
        self.received = False

        self._display = self._empty_display()

        # ── subscribers ─────────────────────────────────────────
        self.create_subscription(
            PointCloud2,
            '/spray/coverage_cloud',
            self._cloud_callback,
            rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            )
        )

        # ── publishers ──────────────────────────────────────────
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.reward_pub    = self.create_publisher(Float32,     '/spray/reward',          10)
        self.unpainted_pub = self.create_publisher(PointCloud2, '/spray/unpainted_cloud', qos)
        self.good_pub      = self.create_publisher(PointCloud2, '/spray/good_cloud',      qos)
        self.overspray_pub = self.create_publisher(PointCloud2, '/spray/overspray_cloud', qos)

        # ── timer ────────────────────────────────────────────────
        self.create_timer(2.0, self._compute_quality)

        # ── matplotlib window ────────────────────────────────────
        self._build_figure()

        self.get_logger().info(
            f'Coverage Quality Node started | '
            f'unpainted<{THR_UNPAINTED} weak<{THR_GOOD_LO} good≤{THR_GOOD_HI} overspray>{THR_GOOD_HI} | '
            f'Nozzle: ST-6 ø1.0mm'
        )

    # ─────────────────────────────────────────────────────────────
    # FIGURE CONSTRUCTION
    # ─────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor('#1c1c1e')

        # Status strip at very top (tall enough to read)
        self.ax_status = self.fig.add_axes([0.0, 0.87, 1.0, 0.13])
        self.ax_status.set_axis_off()
        self.ax_status.set_facecolor('#1c1c1e')

        # Grid: 2 rows × 2 cols below the status strip
        gs = self.fig.add_gridspec(
            2, 2,
            left=0.05, right=0.97,
            top=0.85,  bottom=0.06,
            hspace=0.38, wspace=0.32
        )

        # Top-left: 3D heatmap
        self.ax3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self._style_3d(self.ax3d, '3D Coverage Heatmap')

        # Top-right: 2D Y-Z projection heatmap
        self.ax2d = self.fig.add_subplot(gs[0, 1])
        self._style_2d(self.ax2d, 'Side Projection (Y-Z)  — heatmap')

        # Bottom-left: coverage histogram
        self.ax_hist = self.fig.add_subplot(gs[1, 0])
        self._style_dark(self.ax_hist, 'Coverage Hit-Count Distribution')
        self.ax_hist.set_xlabel('Hit count (coverage)', color='#cccccc')
        self.ax_hist.set_ylabel('Number of voxels',    color='#cccccc')

        # Bottom-right: zone bar chart
        self.ax_bar = self.fig.add_subplot(gs[1, 1])
        self._style_dark(self.ax_bar, 'Zone Breakdown')
        self.ax_bar.set_ylabel('Voxel count', color='#cccccc')

        # Shared colourbar for the two heatmap axes
        sm = cm.ScalarMappable(
            cmap=CMAP,
            norm=mcolors.Normalize(vmin=NORM_VMIN, vmax=NORM_VMAX)
        )
        sm.set_array([])
        cbar_ax = self.fig.add_axes([0.97, 0.50, 0.013, 0.35])
        cbar = self.fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Coverage hits', color='#cccccc', fontsize=8)
        cbar.ax.yaxis.set_tick_params(color='#cccccc')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#cccccc')

        # Add threshold lines on colourbar
        for val, lbl in [(THR_UNPAINTED, 'unpaint.'), (THR_GOOD_LO, 'good↑'), (THR_GOOD_HI, 'over↑')]:
            frac = (val - NORM_VMIN) / (NORM_VMAX - NORM_VMIN)
            cbar.ax.axhline(frac, color='white', linewidth=1.0, linestyle='--')
            cbar.ax.text(1.6, frac, lbl, va='center', ha='left',
                         color='white', fontsize=6, transform=cbar.ax.transAxes)

        self.ani = animation.FuncAnimation(
            self.fig, self._gui_update, interval=600, blit=False
        )

    # ─────────────────────────────────────────────────────────────
    # AXIS STYLE HELPERS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _style_dark(ax, title: str):
        ax.set_facecolor('#2a2a2e')
        ax.tick_params(colors='#cccccc', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
        ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=6)

    def _style_2d(self, ax, title: str):
        self._style_dark(ax, title)
        ax.set_xlabel('Y [m]', color='#cccccc', fontsize=8)
        ax.set_ylabel('Z [m]', color='#cccccc', fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')

    @staticmethod
    def _style_3d(ax, title: str):
        ax.set_facecolor('#2a2a2e')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#444444')
        ax.yaxis.pane.set_edgecolor('#444444')
        ax.zaxis.pane.set_edgecolor('#444444')
        ax.tick_params(colors='#aaaaaa', labelsize=6)
        ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=4)
        ax.set_xlabel('X', color='#aaaaaa', fontsize=7)
        ax.set_ylabel('Y', color='#aaaaaa', fontsize=7)
        ax.set_zlabel('Z', color='#aaaaaa', fontsize=7)

    # ─────────────────────────────────────────────────────────────
    # EMPTY DISPLAY BUFFER
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_display():
        return {
            'xyz':      np.zeros((0, 3), dtype=np.float32),
            'coverage': np.zeros(0,       dtype=np.float32),
            'n_unpainted': 0, 'n_weak': 0, 'n_good': 0, 'n_over': 0,
            'quality_pct': 0.0, 'coverage_pct': 0.0, 'reward': 0.0,
            'ready': False,
        }

    # ─────────────────────────────────────────────────────────────
    # SUBSCRIBER CALLBACK
    # ─────────────────────────────────────────────────────────────

    def _cloud_callback(self, msg: PointCloud2):
        n = msg.width * msg.height
        if n == 0:
            return
        # Vectorised parse — avoids 130k-iteration Python loop that was
        # blocking the spin thread and causing missed messages.
        # coverage_map_node publishes point_step=16 (x,y,z,intensity as float32).
        step = msg.point_step
        raw  = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        # Extract all 4 float32 fields at once using strided indexing
        floats = raw.view(np.float32)
        n_fields = step // 4  # number of float32s per point (should be 4)
        pts = floats[: n * n_fields].reshape(n, n_fields)[:, :4].astype(np.float32)
        with self.lock:
            self.pts      = pts
            self.received = True

    # ─────────────────────────────────────────────────────────────
    # QUALITY COMPUTATION  (2-s timer)
    # ─────────────────────────────────────────────────────────────

    def _compute_quality(self):
        with self.lock:
            if not self.received or len(self.pts) == 0:
                return
            pts = self.pts.copy()

        xyz      = pts[:, :3]
        coverage = pts[:, 3]
        n        = len(coverage)

        unpainted_mask = coverage < THR_UNPAINTED
        weak_mask      = (coverage >= THR_UNPAINTED) & (coverage < self.MIN)
        good_mask      = (coverage >= self.MIN)      & (coverage <= self.MAX)
        over_mask      = coverage > self.MAX

        n_unpainted = int(np.sum(unpainted_mask))
        n_weak      = int(np.sum(weak_mask))
        n_good      = int(np.sum(good_mask))
        n_over      = int(np.sum(over_mask))

        coverage_pct = (np.sum(coverage >= THR_UNPAINTED) / n) * 100.0
        quality_pct  = (n_good / n) * 100.0
        weak_pct     = (n_weak / n) * 100.0
        over_pct     = (n_over / n) * 100.0

        # Reward is purely percentage-based so it doesn't scale with surface size.
        # +2 per % of good coverage (13–38 µm OEM window)
        # -1 per % of weak coverage (below 13 µm — insufficient adhesion)
        # -3 per % of overspray    (above 38 µm — runs/sags risk, material waste)
        reward = quality_pct * 2.0 - weak_pct * 1.0 - over_pct * 3.0

        # ── publish zone clouds ──────────────────────────────────
        stamp    = self.get_clock().now().to_msg()
        frame_id = 'world'
        self.unpainted_pub.publish(self._make_cloud(xyz[unpainted_mask], stamp, frame_id))
        self.good_pub.publish(     self._make_cloud(xyz[good_mask],       stamp, frame_id))
        self.overspray_pub.publish(self._make_cloud(xyz[over_mask],       stamp, frame_id))

        # ── publish reward ───────────────────────────────────────
        rmsg = Float32(); rmsg.data = float(reward)
        self.reward_pub.publish(rmsg)

        # ── warnings ─────────────────────────────────────────────
        for mask, label in [(unpainted_mask, 'Unpainted'),
                            (weak_mask,      'Weak'),
                            (over_mask,      'Overspray')]:
            if mask.any():
                c = xyz[mask].mean(axis=0)
                self.get_logger().warn(
                    f'{label}: {int(mask.sum())} pts | '
                    f'centroid ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})'
                )

        self.get_logger().info(
            f'Quality={quality_pct:.1f}% | Coverage={coverage_pct:.1f}% | '
            f'U={n_unpainted} W={n_weak} G={n_good} O={n_over} | '
            f'Reward={reward:.2f}'
        )

        # ── update display buffer ────────────────────────────────
        with self.lock:
            self._display = {
                'xyz':         xyz,
                'coverage':    coverage,
                'n_unpainted': n_unpainted,
                'n_weak':      n_weak,
                'n_good':      n_good,
                'n_over':      n_over,
                'quality_pct': quality_pct,
                'coverage_pct': coverage_pct,
                'reward':      reward,
                'ready':       True,
            }

    # ─────────────────────────────────────────────────────────────
    # GUI UPDATE  (main thread via FuncAnimation)
    # ─────────────────────────────────────────────────────────────

    def _gui_update(self, frame):
        with self.lock:
            d = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in self._display.items()}

        if not d['ready']:
            return []

        xyz      = d['xyz']
        coverage = d['coverage']
        colours  = _coverage_to_colour(coverage)

        q  = d['quality_pct']
        cv = d['coverage_pct']
        rw = d['reward']

        # ── STATUS STRIP ─────────────────────────────────────────
        self.ax_status.cla()
        self.ax_status.set_axis_off()

        # Traffic-light background based on quality
        if q >= 40:
            bg, fg = '#1a5c1a', '#6dff6d'
        elif q >= 10:
            bg, fg = '#5c4a00', '#ffd54f'
        else:
            bg, fg = '#5c1a1a', '#ff6b6b'

        rect = FancyBboxPatch((0.01, 0.05), 0.98, 0.90,
                               boxstyle='round,pad=0.01',
                               linewidth=2, edgecolor=fg,
                               facecolor=bg,
                               transform=self.ax_status.transAxes,
                               clip_on=False)
        self.ax_status.add_patch(rect)

        status_text = (
            f'GOOD  {q:.1f}%          '
            f'COVERAGE  {cv:.1f}%          '
            f'REWARD  {rw:.1f}'
        )
        self.ax_status.text(
            0.5, 0.50, status_text,
            ha='center', va='center',
            fontsize=18, fontweight='bold',
            color=fg,
            transform=self.ax_status.transAxes,
            family='monospace'
        )

        # Sub-label line
        spray_label = '⬤ SPRAYING' if cv > 0.5 else '○  IDLE — no coverage detected'
        spray_col   = '#6dff6d' if cv > 0.5 else '#ff9999'
        self.ax_status.text(
            0.5, 0.08, spray_label,
            ha='center', va='bottom',
            fontsize=9, color=spray_col,
            transform=self.ax_status.transAxes
        )

        # ── 3D HEATMAP ───────────────────────────────────────────
        self.ax3d.cla()
        self._style_3d(self.ax3d, '3D Coverage Heatmap')

        # Subsample for speed (max 15 k points in 3D)
        idx = np.arange(len(xyz))
        if len(idx) > 15_000:
            idx = np.random.choice(idx, 15_000, replace=False)

        if len(idx):
            self.ax3d.scatter(
                xyz[idx, 0], xyz[idx, 1], xyz[idx, 2],
                c=colours[idx], s=3, alpha=0.75, depthshade=True,
                linewidths=0
            )

        # ── 2D PROJECTION (Y-Z) ──────────────────────────────────
        self.ax2d.cla()
        self._style_2d(self.ax2d, 'Side Projection (Y-Z)  — heatmap')

        if len(xyz):
            # full cloud in 2D (faster than 3D)
            sc = self.ax2d.scatter(
                xyz[:, 1], xyz[:, 2],
                c=coverage, cmap=CMAP,
                vmin=NORM_VMIN, vmax=NORM_VMAX,
                s=1.5, alpha=0.65, linewidths=0
            )

            # Contour boundary: painted vs unpainted
            try:
                from scipy.stats import binned_statistic_2d
                y_range = [xyz[:, 1].min(), xyz[:, 1].max()]
                z_range = [xyz[:, 2].min(), xyz[:, 2].max()]
                res = 80
                stat, ye, ze, _ = binned_statistic_2d(
                    xyz[:, 1], xyz[:, 2], coverage,
                    statistic='mean', bins=res,
                    range=[y_range, z_range]
                )
                yc = 0.5 * (ye[:-1] + ye[1:])
                zc = 0.5 * (ze[:-1] + ze[1:])
                YY, ZZ = np.meshgrid(yc, zc, indexing='ij')
                stat_filled = np.nan_to_num(stat, nan=0.0)
                self.ax2d.contour(
                    YY, ZZ, stat_filled,
                    levels=[THR_UNPAINTED, THR_GOOD_LO, THR_GOOD_HI],
                    colors=['#ffffffaa', '#2dbe4ecc', '#e03030cc'],
                    linewidths=[1.0, 1.2, 1.2],
                    linestyles=['dashed', 'solid', 'solid']
                )
            except Exception:
                pass  # contour is best-effort

            # annotation box
            self.ax2d.text(
                0.02, 0.97,
                f'Painted: {cv:.1f}%  |  Good: {q:.1f}%',
                transform=self.ax2d.transAxes,
                va='top', ha='left', fontsize=8, color='white',
                bbox=dict(facecolor='#00000088', edgecolor='none', pad=3)
            )

        # ── HISTOGRAM ────────────────────────────────────────────
        self.ax_hist.cla()
        self._style_dark(self.ax_hist, 'Coverage Hit-Count Distribution')
        self.ax_hist.set_xlabel('Hit count (coverage)', color='#cccccc', fontsize=8)
        self.ax_hist.set_ylabel('Number of voxels',    color='#cccccc', fontsize=8)

        if len(coverage):
            max_cov = max(float(coverage.max()), THR_GOOD_HI + 2)
            bins    = np.linspace(0, min(max_cov, 30), 40)
            counts, edges = np.histogram(coverage, bins=bins)
            bin_centers   = 0.5 * (edges[:-1] + edges[1:])
            bar_colours   = _coverage_to_colour(bin_centers)

            self.ax_hist.bar(
                bin_centers, counts,
                width=(bins[1] - bins[0]) * 0.9,
                color=bar_colours, alpha=0.9, linewidth=0
            )
            # threshold lines
            for val, lbl, col in [
                (THR_UNPAINTED, 'unpainted', '#ffffff'),
                (THR_GOOD_LO,   'good ↑',   '#2dbe4e'),
                (THR_GOOD_HI,   'over ↑',   '#e03030'),
            ]:
                self.ax_hist.axvline(val, color=col, linewidth=1.5, linestyle='--', alpha=0.85)
                self.ax_hist.text(
                    val + 0.2, self.ax_hist.get_ylim()[1] * 0.92,
                    lbl, color=col, fontsize=7, va='top'
                )

        # ── ZONE BAR ─────────────────────────────────────────────
        self.ax_bar.cla()
        self._style_dark(self.ax_bar, 'Zone Breakdown')
        self.ax_bar.set_ylabel('Voxel count', color='#cccccc', fontsize=8)

        zones  = ['Unpainted', 'Weak', 'Good', 'Overspray']
        counts = [d['n_unpainted'], d['n_weak'], d['n_good'], d['n_over']]
        cols   = [C_UNPAINTED,     C_WEAK,      C_GOOD,      C_OVER]
        bars   = self.ax_bar.bar(zones, counts, color=cols, alpha=0.9, width=0.55)
        total  = max(sum(counts), 1)
        for bar, cnt in zip(bars, counts):
            pct = cnt / total * 100
            self.ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f'{pct:.1f}%',
                ha='center', va='bottom',
                color='white', fontsize=8, fontweight='bold'
            )
        self.ax_bar.tick_params(colors='#cccccc', labelsize=8)

        return []

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_cloud(xyz: np.ndarray, stamp, frame_id: str) -> PointCloud2:
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        n          = len(xyz)
        point_step = 12
        data       = bytearray(n * point_step)
        for i, pt in enumerate(xyz):
            struct.pack_into('fff', data, i * point_step,
                             float(pt[0]), float(pt[1]), float(pt[2]))
        msg = PointCloud2()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.height          = 1
        msg.width           = n
        msg.fields          = fields
        msg.is_bigendian    = False
        msg.point_step      = point_step
        msg.row_step        = point_step * n
        msg.data            = bytes(data)
        msg.is_dense        = True
        return msg

    def destroy_node(self):
        plt.close('all')
        super().destroy_node()


# =============================================================
# MAIN
# =============================================================

def main(args=None):
    rclpy.init(args=args)
    node = CoverageQualityNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()