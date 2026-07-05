#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import rclpy.qos

import numpy as np
import threading
import random
import math
import os
import collections

from std_msgs.msg import Float32, Float32MultiArray, String, Bool, Float64
from geometry_msgs.msg import Point, PoseStamped, Pose
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

from scipy.ndimage import gaussian_filter

try:
    from sklearn.cluster import DBSCAN
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False


# ═══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════

def _orientation_facing_normal(nx, ny, nz):
    """Quaternion (x,y,z,w) whose tool-Z points along (nx,ny,nz)."""
    tool_z = np.array([nx, ny, nz], dtype=np.float64)
    n = np.linalg.norm(tool_z)
    if n < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    tool_z /= n

    ref = np.array([0.0, 0.0, 1.0])
    if abs(tool_z @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])

    tool_y = np.cross(tool_z, ref); tool_y /= np.linalg.norm(tool_y)
    tool_x = np.cross(tool_y, tool_z); tool_x /= np.linalg.norm(tool_x)

    R = np.stack([tool_x, tool_y, tool_z], axis=1)
    t = R[0,0] + R[1,1] + R[2,2]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s; x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s; z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s; x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s; z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s; x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s; z = 0.25 * s
    n2 = math.sqrt(x*x + y*y + z*z + w*w)
    return x/n2, y/n2, z/n2, w/n2


# ═══════════════════════════════════════════════════════════════
#  NEURAL NETWORK PRIMITIVES  (pure numpy, no framework dep.)
# ═══════════════════════════════════════════════════════════════

def _relu(x):   return np.maximum(0.0, x)
def _tanh(x):   return np.tanh(x)
def _softplus(x, beta=1.0): return np.log1p(np.exp(np.clip(beta * x, -30, 30))) / beta

def _layer_forward(x, W, b, activation='relu'):
    y = x @ W.T + b
    if activation == 'relu':  return _relu(y)
    if activation == 'tanh':  return _tanh(y)
    if activation == 'linear': return y
    return y

def _init_layer(fan_in, fan_out, scale=None):
    if scale is None:
        scale = math.sqrt(2.0 / fan_in)   # He init
    W = np.random.randn(fan_out, fan_in).astype(np.float32) * scale
    b = np.zeros(fan_out, dtype=np.float32)
    return W, b


# ═══════════════════════════════════════════════════════════════
#  DIMENSIONS & HYPER-PARAMETERS
# ═══════════════════════════════════════════════════════════════

OBS_DIM  = 8
ACT_DIM  = 4
HIDDEN   = 64

# Physical limits — mirror cartesian_trajectory_controller.cpp
STANDOFF_MIN = 0.15
STANDOFF_MAX = 0.25   # matches cartesian_trajectory_controller.cpp STANDOFF_MAX clamp

# Thickness classification thresholds
MIN_THICKNESS          = 25.0
MAX_THICKNESS          = 75.0
UNEVEN_GRADIENT_THRESH = 400.0

# DBSCAN
DBSCAN_EPS         = 3
DBSCAN_MIN_SAMPLES = 4

# Patch dimensions for the boustrophedon path (metres)
PATCH_HALF_WIDTH = 0.15   # half-size of correction patch in Y
PATCH_HALF_HEIGHT= 0.15   # half-size of correction patch in Z
PATCH_STEP       = 0.02   # row spacing

# Decision rate
DECISION_INTERVAL = 3.0   # seconds

# Default surface normal — overridden at runtime from tracking pose / CSV geometry.

DEFAULT_SURFACE_NX = 1.0
DEFAULT_SURFACE_NY = 0.0
DEFAULT_SURFACE_NZ = 0.0

# ── PPO ──────────────────────────────────────────────────────
PPO_LR          = 3e-4
PPO_CLIP_EPS    = 0.2
PPO_GAMMA       = 0.99
PPO_LAM         = 0.95     # GAE lambda
PPO_EPOCHS      = 4        # mini-epochs per update
PPO_MINI_BATCH  = 16
PPO_ROLLOUT_LEN = 32       # steps collected before each update
PPO_ENT_COEF    = 0.01     # entropy bonus
PPO_VF_COEF     = 0.5
PPO_MAX_GRAD    = 0.5
LOG_STD_MIN     = -4.0
LOG_STD_MAX     = 1.0

# ── TD3 ──────────────────────────────────────────────────────
TD3_LR              = 1e-3
TD3_GAMMA           = 0.99
TD3_TAU             = 0.005    # soft target update
TD3_POLICY_NOISE    = 0.2
TD3_NOISE_CLIP      = 0.5
TD3_POLICY_DELAY    = 2        # update actor every N critic updates
TD3_BUFFER_SIZE     = 50_000
TD3_BATCH_SIZE      = 64
TD3_WARMUP_STEPS    = 200      # random actions until buffer has this many
TD3_MAX_BLEND       = 0.5      # max weight of TD3 in ensemble

# Paths for persistence
CHECKPOINT_DIR = "/home/user/car_spraying_ws/rl_checkpoints"


# ═══════════════════════════════════════════════════════════════
#  PPO AGENT  (actor-critic, Gaussian policy)
# ═══════════════════════════════════════════════════════════════

class PPOAgent:
    """
    Actor-critic PPO with a shared trunk and separate heads.
    Fully numpy — runs on CPU without any ML framework.

    Actor outputs mean + log_std per action dimension.
    Critic outputs a scalar value estimate.
    GAE advantage estimation.  Clipped surrogate objective.
    """

    def __init__(self):
        # Shared trunk: OBS_DIM → HIDDEN → HIDDEN
        self.W_s1, self.b_s1 = _init_layer(OBS_DIM, HIDDEN)
        self.W_s2, self.b_s2 = _init_layer(HIDDEN,  HIDDEN)

        # Actor head: HIDDEN → ACT_DIM (mean)
        self.W_mu, self.b_mu = _init_layer(HIDDEN, ACT_DIM, scale=0.01)
        # Log-std: learned scalar per action dim (not input-dependent)
        self.log_std = np.zeros(ACT_DIM, dtype=np.float32)

        # Critic head: HIDDEN → 1
        self.W_v, self.b_v = _init_layer(HIDDEN, 1, scale=1.0)

        # Rollout buffer
        self._obs   = []
        self._acts  = []
        self._rews  = []
        self._vals  = []
        self._lps   = []
        self._dones = []

        self._step = 0

    def _trunk(self, obs):
        h = _layer_forward(obs, self.W_s1, self.b_s1, 'relu')
        h = _layer_forward(h,   self.W_s2, self.b_s2, 'relu')
        return h

    def _value(self, obs):
        h = self._trunk(np.atleast_2d(obs))
        return float((h @ self.W_v.T + self.b_v).squeeze())

    def _policy(self, obs):
        """Returns (mean, std) numpy arrays of shape (ACT_DIM,)."""
        h   = self._trunk(np.atleast_2d(obs))
        mu  = (h @ self.W_mu.T + self.b_mu).squeeze()
        std = np.exp(np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
        return mu.astype(np.float32), std.astype(np.float32)

    def select_action(self, obs):
        """Sample action, return (action_clipped, log_prob, value)."""
        mu, std = self._policy(obs)
        raw  = mu + std * np.random.randn(ACT_DIM).astype(np.float32)
        # log prob of Gaussian
        lp   = -0.5 * np.sum(((raw - mu) / (std + 1e-8))**2 + 2 * np.log(std + 1e-8))
        val  = self._value(obs)
        return raw, float(lp), float(val)

    def store(self, obs, act, rew, val, lp, done):
        self._obs.append(obs.copy())
        self._acts.append(act.copy())
        self._rews.append(rew)
        self._vals.append(val)
        self._lps.append(lp)
        self._dones.append(done)
        self._step += 1

    def ready(self):
        return self._step >= PPO_ROLLOUT_LEN

    def update(self, last_obs, last_done):
        """Run PPO update on the collected rollout. Returns mean policy loss."""
        # Bootstrap value
        last_val = 0.0 if last_done else self._value(last_obs)

        # GAE
        adv = np.zeros(len(self._rews), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(self._rews))):
            nv  = self._vals[t+1] if t+1 < len(self._vals) else last_val
            nd  = self._dones[t]
            delta = self._rews[t] + PPO_GAMMA * nv * (1 - nd) - self._vals[t]
            gae   = delta + PPO_GAMMA * PPO_LAM * (1 - nd) * gae
            adv[t] = gae
        rets = adv + np.array(self._vals, dtype=np.float32)
        adv  = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_b  = np.array(self._obs,  dtype=np.float32)
        act_b  = np.array(self._acts, dtype=np.float32)
        lp_old = np.array(self._lps,  dtype=np.float32)

        total_loss = 0.0
        n = len(obs_b)

        for _ in range(PPO_EPOCHS):
            idx = np.random.permutation(n)
            for start in range(0, n, PPO_MINI_BATCH):
                mb = idx[start:start + PPO_MINI_BATCH]
                if len(mb) < 2:
                    continue

                o_mb  = obs_b[mb]
                a_mb  = act_b[mb]
                adv_mb = adv[mb]
                ret_mb = rets[mb]
                lp_mb  = lp_old[mb]

                # Forward
                h = _relu(o_mb @ self.W_s1.T + self.b_s1)
                h = _relu(h    @ self.W_s2.T + self.b_s2)
                mu_mb  = h @ self.W_mu.T + self.b_mu
                std_mb = np.exp(np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
                val_mb = (h @ self.W_v.T + self.b_v).squeeze()

                # New log probs
                lp_new = -0.5 * np.sum(
                    ((a_mb - mu_mb) / (std_mb + 1e-8))**2
                    + 2 * np.log(std_mb + 1e-8), axis=1)

                ratio = np.exp(lp_new - lp_mb)
                clip_r = np.clip(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS)
                policy_loss = -np.mean(np.minimum(ratio * adv_mb, clip_r * adv_mb))

                value_loss  = np.mean((val_mb - ret_mb)**2)
                entropy     = np.mean(0.5 * (1 + np.log(2 * math.pi * std_mb**2 + 1e-8)))
                loss = policy_loss + PPO_VF_COEF * value_loss - PPO_ENT_COEF * entropy
                total_loss += loss

                # ── Backward (manual gradients) ──────────────────────────
                # Value head gradient
                dval = 2.0 * PPO_VF_COEF * (val_mb - ret_mb) / len(mb)
                if val_mb.ndim == 0:
                    dval_col = np.array([[dval]])
                else:
                    dval_col = dval[:, None]
                dW_v = dval_col.T @ h / len(mb)
                db_v = dval_col.mean(axis=0).squeeze()

                # Policy head gradient (surrogate)
                # d(loss)/d(mu) via ratio * adv
                surr_mask = (ratio * adv_mb < clip_r * adv_mb).astype(np.float32)
                d_lp = -surr_mask * adv_mb / len(mb)
                d_mu = d_lp[:, None] * (-(a_mb - mu_mb) / (std_mb**2 + 1e-8))
                dW_mu = d_mu.T @ h / len(mb)
                db_mu = d_mu.mean(axis=0)

                # Trunk gradient
                d_h2  = d_mu @ self.W_mu + dval_col * self.W_v
                d_h2 *= (h > 0).astype(np.float32)    # ReLU mask (layer 2)
                dW_s2 = d_h2.T @ _relu(o_mb @ self.W_s1.T + self.b_s1) / len(mb)
                db_s2 = d_h2.mean(axis=0)

                h1 = _relu(o_mb @ self.W_s1.T + self.b_s1)
                d_h1 = d_h2 @ self.W_s2
                d_h1 *= (h1 > 0).astype(np.float32)
                dW_s1 = d_h1.T @ o_mb / len(mb)
                db_s1 = d_h1.mean(axis=0)

                # Log-std gradient (entropy + policy)
                d_log_std = d_lp[:, None] * (
                    (a_mb - mu_mb)**2 / (std_mb**3 + 1e-8) - 1.0 / (std_mb + 1e-8))
                d_log_std -= PPO_ENT_COEF * (std_mb / (std_mb**2 + 1e-8))
                d_log_std  = d_log_std.mean(axis=0)

                # Gradient clipping
                def _clip(g):
                    n = np.linalg.norm(g)
                    return g if n <= PPO_MAX_GRAD else g * PPO_MAX_GRAD / n

                self.W_s1  -= PPO_LR * _clip(dW_s1)
                self.b_s1  -= PPO_LR * _clip(db_s1)
                self.W_s2  -= PPO_LR * _clip(dW_s2)
                self.b_s2  -= PPO_LR * _clip(db_s2)
                self.W_mu  -= PPO_LR * _clip(dW_mu)
                self.b_mu  -= PPO_LR * _clip(db_mu)
                self.W_v   -= PPO_LR * _clip(dW_v.reshape(self.W_v.shape))
                self.b_v   -= PPO_LR * _clip(np.atleast_1d(db_v))
                self.log_std -= PPO_LR * _clip(d_log_std)
                self.log_std  = np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)

        # Clear buffer
        self._obs.clear(); self._acts.clear(); self._rews.clear()
        self._vals.clear(); self._lps.clear(); self._dones.clear()
        self._step = 0

        return total_loss

    def save(self, path):
        np.savez(path,
            W_s1=self.W_s1, b_s1=self.b_s1,
            W_s2=self.W_s2, b_s2=self.b_s2,
            W_mu=self.W_mu, b_mu=self.b_mu,
            W_v=self.W_v,   b_v=self.b_v,
            log_std=self.log_std)

    def load(self, path):
        d = np.load(path)
        self.W_s1 = d['W_s1']; self.b_s1 = d['b_s1']
        self.W_s2 = d['W_s2']; self.b_s2 = d['b_s2']
        self.W_mu = d['W_mu']; self.b_mu = d['b_mu']
        self.W_v  = d['W_v'];  self.b_v  = d['b_v']
        self.log_std = d['log_std']


# ═══════════════════════════════════════════════════════════════
#  TD3 AGENT  (Twin Delayed DDPG)
# ═══════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, maxlen=TD3_BUFFER_SIZE):
        self._buf = collections.deque(maxlen=maxlen)

    def add(self, obs, act, rew, next_obs, done):
        self._buf.append((obs.copy(), act.copy(), rew,
                          next_obs.copy(), float(done)))

    def sample(self, n):
        batch = random.sample(self._buf, min(n, len(self._buf)))
        obs, acts, rews, nobs, dones = zip(*batch)
        return (np.array(obs,   dtype=np.float32),
                np.array(acts,  dtype=np.float32),
                np.array(rews,  dtype=np.float32),
                np.array(nobs,  dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self): return len(self._buf)


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient.
    Deterministic actor, twin critics, delayed policy updates.
    Pure numpy implementation.
    """

    def __init__(self):
        # Actor: OBS → HIDDEN → HIDDEN → ACT (tanh output)
        self.aW1, self.ab1 = _init_layer(OBS_DIM, HIDDEN)
        self.aW2, self.ab2 = _init_layer(HIDDEN,  HIDDEN)
        self.aW3, self.ab3 = _init_layer(HIDDEN,  ACT_DIM, scale=0.01)

        # Actor target (copy)
        self.taW1, self.tab1 = self.aW1.copy(), self.ab1.copy()
        self.taW2, self.tab2 = self.aW2.copy(), self.ab2.copy()
        self.taW3, self.tab3 = self.aW3.copy(), self.ab3.copy()

        # Critic 1: (OBS + ACT) → HIDDEN → HIDDEN → 1
        inp = OBS_DIM + ACT_DIM
        self.c1W1, self.c1b1 = _init_layer(inp,    HIDDEN)
        self.c1W2, self.c1b2 = _init_layer(HIDDEN, HIDDEN)
        self.c1W3, self.c1b3 = _init_layer(HIDDEN, 1, scale=1.0)

        # Critic 2
        self.c2W1, self.c2b1 = _init_layer(inp,    HIDDEN)
        self.c2W2, self.c2b2 = _init_layer(HIDDEN, HIDDEN)
        self.c2W3, self.c2b3 = _init_layer(HIDDEN, 1, scale=1.0)

        # Target critics
        for attr in ['c1W1','c1b1','c1W2','c1b2','c1W3','c1b3',
                     'c2W1','c2b1','c2W2','c2b2','c2W3','c2b3']:
            setattr(self, 't'+attr, getattr(self, attr).copy())

        self._update_count = 0
        self.buffer = ReplayBuffer()

    def _actor_forward(self, obs, W1, b1, W2, b2, W3, b3):
        h = _relu(obs @ W1.T + b1)
        h = _relu(h   @ W2.T + b2)
        return _tanh(h @ W3.T + b3)

    def _critic_forward(self, obs, act, W1, b1, W2, b2, W3, b3):
        x = np.concatenate([obs, act], axis=-1)
        h = _relu(x @ W1.T + b1)
        h = _relu(h @ W2.T + b2)
        return (h @ W3.T + b3).squeeze()

    def select_action(self, obs, noise=0.0):
        """Deterministic action in [-1, 1]^ACT_DIM, optional exploration noise."""
        obs2 = np.atleast_2d(obs).astype(np.float32)
        act  = self._actor_forward(obs2,
                    self.aW1, self.ab1, self.aW2, self.ab2, self.aW3, self.ab3)
        act  = act.squeeze().astype(np.float32)
        if noise > 0.0:
            act += noise * np.random.randn(ACT_DIM).astype(np.float32)
        return np.clip(act, -1.0, 1.0)

    def _soft_update(self, src, tgt_name):
        """Polyak average: target ← τ*src + (1-τ)*target"""
        t = getattr(self, tgt_name)
        setattr(self, tgt_name, TD3_TAU * src + (1 - TD3_TAU) * t)

    def update(self):
        """One TD3 update step. Returns (critic_loss, actor_loss or None)."""
        if len(self.buffer) < TD3_BATCH_SIZE:
            return None, None

        obs, acts, rews, nobs, dones = self.buffer.sample(TD3_BATCH_SIZE)

        # ── Target actions with clipped noise ────────────────────────────
        noise = np.clip(
            TD3_POLICY_NOISE * np.random.randn(TD3_BATCH_SIZE, ACT_DIM),
            -TD3_NOISE_CLIP, TD3_NOISE_CLIP).astype(np.float32)
        t_acts = np.clip(
            self._actor_forward(nobs,
                self.taW1, self.tab1, self.taW2, self.tab2, self.taW3, self.tab3)
            + noise, -1.0, 1.0)

        # ── Target Q ─────────────────────────────────────────────────────
        tq1 = self._critic_forward(nobs, t_acts,
            self.tc1W1, self.tc1b1, self.tc1W2, self.tc1b2, self.tc1W3, self.tc1b3)
        tq2 = self._critic_forward(nobs, t_acts,
            self.tc2W1, self.tc2b1, self.tc2W2, self.tc2b2, self.tc2W3, self.tc2b3)
        target_q = rews + TD3_GAMMA * (1 - dones) * np.minimum(tq1, tq2)

        # ── Critic update (SGD step on MSE) ──────────────────────────────
        def _critic_loss_grad(W1, b1, W2, b2, W3, b3, target):
            x  = np.concatenate([obs, acts], axis=-1)
            h1 = _relu(x  @ W1.T + b1)
            h2 = _relu(h1 @ W2.T + b2)
            q  = (h2 @ W3.T + b3).squeeze()
            err = (q - target) / len(target)
            d3  = err[:, None]
            dW3 = d3.T @ h2; db3 = d3.mean(axis=0).squeeze()
            d2  = d3 * W3; d2 *= (h2 > 0)
            dW2 = d2.T @ h1; db2 = d2.mean(axis=0)
            d1  = d2 @ W2; d1 *= (h1 > 0)
            dW1 = d1.T @ x; db1 = d1.mean(axis=0)
            loss = float(np.mean(err**2))
            return loss, dW1, db1, dW2, db2, dW3, db3

        cl1, g1W1,g1b1,g1W2,g1b2,g1W3,g1b3 = _critic_loss_grad(
            self.c1W1,self.c1b1,self.c1W2,self.c1b2,self.c1W3,self.c1b3, target_q)
        cl2, g2W1,g2b1,g2W2,g2b2,g2W3,g2b3 = _critic_loss_grad(
            self.c2W1,self.c2b1,self.c2W2,self.c2b2,self.c2W3,self.c2b3, target_q)

        lr = TD3_LR
        self.c1W1 -= lr*g1W1; self.c1b1 -= lr*g1b1
        self.c1W2 -= lr*g1W2; self.c1b2 -= lr*g1b2
        self.c1W3 -= lr*g1W3.reshape(self.c1W3.shape); self.c1b3 -= lr*np.atleast_1d(g1b3)
        self.c2W1 -= lr*g2W1; self.c2b1 -= lr*g2b1
        self.c2W2 -= lr*g2W2; self.c2b2 -= lr*g2b2
        self.c2W3 -= lr*g2W3.reshape(self.c2W3.shape); self.c2b3 -= lr*np.atleast_1d(g2b3)

        self._update_count += 1
        actor_loss = None

        # ── Delayed actor update ──────────────────────────────────────────
        if self._update_count % TD3_POLICY_DELAY == 0:
            a_pred = self._actor_forward(obs,
                self.aW1, self.ab1, self.aW2, self.ab2, self.aW3, self.ab3)
            # Actor loss = -mean(Q1(s, π(s)))
            x  = np.concatenate([obs, a_pred], axis=-1)
            h1 = _relu(x  @ self.c1W1.T + self.c1b1)
            h2 = _relu(h1 @ self.c1W2.T + self.c1b2)
            q  = (h2 @ self.c1W3.T + self.c1b3).squeeze()
            actor_loss = float(-np.mean(q))

            # dQ/da via chain rule back through actor
            dq_dh2 = np.ones((len(obs), 1)) * self.c1W3 / len(obs)
            dh2_dh1 = (h2 > 0).astype(np.float32)
            dq_dh1  = (dq_dh2 * dh2_dh1) @ self.c1W2
            dh1_dx  = (h1 > 0).astype(np.float32)
            dq_dx   = (dq_dh1 * dh1_dx) @ self.c1W1
            dq_da   = -dq_dx[:, OBS_DIM:]   # only action part

            # Actor backward
            h1a = _relu(obs @ self.aW1.T + self.ab1)
            h2a = _relu(h1a @ self.aW2.T + self.ab2)
            out_a = _tanh(h2a @ self.aW3.T + self.ab3)
            d_tanh = 1.0 - out_a**2
            d3a   = dq_da * d_tanh
            dW3a  = d3a.T @ h2a; db3a = d3a.mean(axis=0)
            d2a   = d3a @ self.aW3 * (h2a > 0)
            dW2a  = d2a.T @ h1a; db2a = d2a.mean(axis=0)
            d1a   = d2a @ self.aW2 * (h1a > 0)
            dW1a  = d1a.T @ obs; db1a = d1a.mean(axis=0)

            self.aW1 -= lr*dW1a; self.ab1 -= lr*db1a
            self.aW2 -= lr*dW2a; self.ab2 -= lr*db2a
            self.aW3 -= lr*dW3a.reshape(self.aW3.shape); self.ab3 -= lr*db3a

            # Soft target updates
            for src, tname in [
                (self.aW1,'taW1'),(self.ab1,'tab1'),
                (self.aW2,'taW2'),(self.ab2,'tab2'),
                (self.aW3,'taW3'),(self.ab3,'tab3'),
                (self.c1W1,'tc1W1'),(self.c1b1,'tc1b1'),
                (self.c1W2,'tc1W2'),(self.c1b2,'tc1b2'),
                (self.c1W3,'tc1W3'),(self.c1b3,'tc1b3'),
                (self.c2W1,'tc2W1'),(self.c2b1,'tc2b1'),
                (self.c2W2,'tc2W2'),(self.c2b2,'tc2b2'),
                (self.c2W3,'tc2W3'),(self.c2b3,'tc2b3'),
            ]:
                self._soft_update(src, tname)

        return float((cl1 + cl2) / 2.0), actor_loss

    def save(self, path):
        np.savez(path,
            aW1=self.aW1, ab1=self.ab1, aW2=self.aW2, ab2=self.ab2,
            aW3=self.aW3, ab3=self.ab3,
            c1W1=self.c1W1, c1b1=self.c1b1, c1W2=self.c1W2, c1b2=self.c1b2,
            c1W3=self.c1W3, c1b3=self.c1b3,
            c2W1=self.c2W1, c2b1=self.c2b1, c2W2=self.c2W2, c2b2=self.c2b2,
            c2W3=self.c2W3, c2b3=self.c2b3)

    def load(self, path):
        d = np.load(path)
        for k in d.files:
            setattr(self, k, d[k])
        # Sync targets from loaded weights
        for attr in ['aW1','ab1','aW2','ab2','aW3','ab3',
                     'c1W1','c1b1','c1W2','c1b2','c1W3','c1b3',
                     'c2W1','c2b1','c2W2','c2b2','c2W3','c2b3']:
            setattr(self, 't'+attr, getattr(self, attr).copy())


# ═══════════════════════════════════════════════════════════════
#  ACTION DECODING  (normalised [-1,1] → physical units)
# ═══════════════════════════════════════════════════════════════

def decode_action(raw_action, y_min, y_max, z_min, z_max):
    """
    Map raw policy output (any range) to physical parameters.
    Returns dict with keys: standoff, flow, target_y, target_z
    """
    a = np.clip(raw_action, -1.0, 1.0)
    # Linear maps from [-1,1]:
    standoff = STANDOFF_MIN + (a[0] + 1.0) * 0.5 * (STANDOFF_MAX - STANDOFF_MIN)
    flow     = np.clip((a[1] + 1.0) * 0.5, 0.0, 1.0)
    ty       = y_min + (a[2] + 1.0) * 0.5 * (y_max - y_min)
    tz       = z_min + (a[3] + 1.0) * 0.5 * (z_max - z_min)
    return dict(standoff=float(standoff), flow=float(flow),
                target_y=float(ty), target_z=float(tz))


# ═══════════════════════════════════════════════════════════════
#  RL AGENT NODE
# ═══════════════════════════════════════════════════════════════

class RLAgentNode(Node):

    def __init__(self):
        super().__init__('rl_agent')

        # ── Grid parameters ──────────────────────────────────────────────
        self.declare_parameter('y_min',      -0.20)
        self.declare_parameter('y_max',       0.00)
        self.declare_parameter('z_min',       0.45)
        self.declare_parameter('z_max',       0.70)
        self.declare_parameter('resolution',  0.02)

        self.y_min = self.get_parameter('y_min').value
        self.y_max = self.get_parameter('y_max').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        self.res   = self.get_parameter('resolution').value

        self.grid_cols = int(np.ceil((self.y_max - self.y_min) / self.res))
        self.grid_rows = int(np.ceil((self.z_max - self.z_min) / self.res))

        # ── Internal state ───────────────────────────────────────────────
        self.thickness_grid = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=np.float32)
        self.lock  = threading.Lock()
        self._pass1_done = False

        # ── EE tracking ─────────────────────────────────────────────────
        # Latest EE pose received from /spray/tracking_pose (cartesian_trajectory_controller.cpp)
        self._current_ee_pose: Pose | None = None
        self._tracking_lock = threading.Lock()

        # ── RL agents ────────────────────────────────────────────────────
        self.ppo = PPOAgent()
        self.td3 = TD3Agent()

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self._try_load_checkpoints()

        self._prev_obs    = None
        self._prev_action = None
        self._episode_step = 0
        self._total_reward = 0.0

        # ── Singularity / manipulability health ─────────────────────────────
        # cartesian_trajectory_controller.cpp ALREADY publishes these two topics every time it
        # runs the Jacobian-SVD singularity check (see trajectory_is_singular
        # in cartesian_trajectory_controller.cpp) — but nothing was subscribing to them before,
        # so the RL agent had no idea when its proposed correction had been
        # rejected for being near-singular. This closes that loop.
        #   /singularity_warning (Bool)    -> True the moment a segment gets
        #                                      rejected outright
        #   /manipulability (Float64)      -> smallest Jacobian singular
        #                                      value seen; cartesian_trajectory_controller.cpp's
        #                                      own threshold is 0.01
        self._MANIP_THRESHOLD = 0.01   # mirrors MANIP_THRESHOLD in cartesian_trajectory_controller.cpp
        self._last_manipulability = None
        self._singularity_seen = False
        self._ik_penalty = 0.0

        # ── Planning failure feedback ────────────────────────────────────────
        # cartesian_trajectory_controller.cpp now publishes /spray/planning_failed=True whenever
        # MoveIt could not plan or execute a requested motion for a reason
        # OTHER than the singularity check above (unreachable target from
        # setPoseTarget()+plan(), rejected execution, etc). Without this,
        # the RL agent could keep re-choosing an out-of-workspace target
        # every decision step and never learn why nothing painted.
        self._planning_failed_seen = False
        self._planning_penalty = 0.0

        # ── QoS ──────────────────────────────────────────────────────────
        _tl_qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )

        # ── Subscribers ──────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, '/spray/thickness_matrix',
            self._matrix_cb, rclpy.qos.qos_profile_sensor_data)

        self.create_subscription(
            Bool, '/spray/pass1_done',
            self._pass1_done_cb, _tl_qos)

        self.create_subscription(
            PoseStamped, '/spray/tracking_pose',
            self._tracking_pose_cb,
            rclpy.qos.qos_profile_sensor_data)

        # These are already published live by cartesian_trajectory_controller.cpp — see
        # trajectory_is_singular() there. No C++ changes needed.
        self.create_subscription(
            Bool, '/singularity_warning',
            self._singularity_cb,
            rclpy.qos.qos_profile_sensor_data)

        self.create_subscription(
            Float64, '/manipulability',
            self._manipulability_cb,
            rclpy.qos.qos_profile_sensor_data)

        # Fires on any MoveIt planning/execution failure that ISN'T a
        # singularity (unreachable pose, rejected IK, etc) — see
        # execute_rl_path() / drain_rl_corrections() in cartesian_trajectory_controller.cpp.
        self.create_subscription(
            Bool, '/spray/planning_failed',
            self._planning_failed_cb,
            rclpy.qos.qos_profile_sensor_data)

        # ── Publishers ───────────────────────────────────────────────────
        self.action_pub = self.create_publisher(Float32MultiArray, '/spray/rl_action', 10)
        self.path_pub   = self.create_publisher(Path,             '/spray/rl_path',   10)
        self.enable_pub = self.create_publisher(Bool,             '/spray/enable',    _tl_qos)
        self.target_pub = self.create_publisher(Point,            '/spray/rl_target', 10)
        self.reward_pub = self.create_publisher(Float32,          '/spray/reward',    10)
        self.status_pub = self.create_publisher(String,           '/spray/rl_status', 10)
        self.marker_pub = self.create_publisher(MarkerArray,      '/spray/defect_markers', 10)

        self.create_timer(DECISION_INTERVAL, self._decision_step)

        # Ensure spray starts OFF during PASS 1
        self._set_spray(False)

        self.get_logger().info(
            f'RL Agent (PPO+TD3) started | '
            f'grid={self.grid_rows}×{self.grid_cols} | '
            f'sklearn={_HAVE_SKLEARN} | '
            f'Waiting for /spray/pass1_done'
        )

    # ─────────────────────────────────────────────────────────
    #  CHECKPOINTING
    # ─────────────────────────────────────────────────────────

    def _try_load_checkpoints(self):
        ppo_path = os.path.join(CHECKPOINT_DIR, 'ppo.npz')
        td3_path = os.path.join(CHECKPOINT_DIR, 'td3.npz')
        if os.path.exists(ppo_path):
            try:
                self.ppo.load(ppo_path)
                self.get_logger().info(f'PPO checkpoint loaded from {ppo_path}')
            except Exception as e:
                self.get_logger().warning(f'PPO load failed: {e}')
        if os.path.exists(td3_path):
            try:
                self.td3.load(td3_path)
                self.get_logger().info(f'TD3 checkpoint loaded from {td3_path}')
            except Exception as e:
                self.get_logger().warning(f'TD3 load failed: {e}')

    def _save_checkpoints(self):
        try:
            self.ppo.save(os.path.join(CHECKPOINT_DIR, 'ppo.npz'))
            self.td3.save(os.path.join(CHECKPOINT_DIR, 'td3.npz'))
        except Exception as e:
            self.get_logger().warning(f'Checkpoint save failed: {e}')

    # ─────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────

    def _tracking_pose_cb(self, msg: PoseStamped):
        """Store the latest EE pose for spray-on/off decisions."""
        with self._tracking_lock:
            self._current_ee_pose = msg.pose

    def _matrix_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        try:
            reshaped = data.reshape((self.grid_rows, self.grid_cols))
            with self.lock:
                self.thickness_grid = reshaped
        except Exception as e:
            self.get_logger().error(f'Thickness reshape failed: {e}')

    def _pass1_done_cb(self, msg: Bool):
        if msg.data and not self._pass1_done:
            self._pass1_done = True
            self.get_logger().info('PASS 1 complete — RL agent now active.')

    def _singularity_cb(self, msg: Bool):
        # Sticky within a decision interval — trajectory_is_singular() in
        # cartesian_trajectory_controller.cpp fires many times per corrective segment; we only
        # need to know if ANY of them tripped since our last decision step.
        if msg.data:
            self._singularity_seen = True

    def _planning_failed_cb(self, msg: Bool):
        # Sticky within a decision interval, same pattern as
        # _singularity_cb — cartesian_trajectory_controller.cpp may fire this several times
        # per decision step (approach failure, then execution failure).
        if msg.data:
            self._planning_failed_seen = True

    def _manipulability_cb(self, msg: Float64):
        # Track the worst (smallest) manipulability seen since the last
        # decision step, since that's what cartesian_trajectory_controller.cpp actually gates on.
        if self._last_manipulability is None:
            self._last_manipulability = float(msg.data)
        else:
            self._last_manipulability = min(self._last_manipulability, float(msg.data))

    def _consume_ik_penalty(self) -> float:
        penalty = 0.0
        if self._singularity_seen:
            penalty = 100.0
        elif self._last_manipulability is not None:
            warn_zone = 5.0 * self._MANIP_THRESHOLD   # 0.05
            if self._last_manipulability < warn_zone:
                frac = (warn_zone - self._last_manipulability) / warn_zone
                penalty = 30.0 * float(np.clip(frac, 0.0, 1.0))

        self._singularity_seen = False
        self._last_manipulability = None
        return penalty

    def _consume_planning_penalty(self) -> float:
        penalty = 150.0 if self._planning_failed_seen else 0.0
        self._planning_failed_seen = False
        return penalty

    # ─────────────────────────────────────────────────────────
    #  OBSERVATION BUILDER
    # ─────────────────────────────────────────────────────────

    def _build_obs(self, smoothed: np.ndarray) -> np.ndarray:
        """Build the 8-dimensional observation vector."""
        total = smoothed.size

        unpainted = float(np.sum(smoothed < 1.0)) / total
        weak      = float(np.sum((smoothed >= 1.0) & (smoothed < MIN_THICKNESS))) / total
        good      = float(np.sum((smoothed >= MIN_THICKNESS) & (smoothed <= MAX_THICKNESS))) / total
        over      = float(np.sum(smoothed > MAX_THICKNESS)) / total

        gy, gx  = np.gradient(smoothed)
        grad_mag = np.sqrt(gx**2 + gy**2)
        uneven   = float(np.sum(grad_mag > UNEVEN_GRADIENT_THRESH)) / total

        mean_t = float(np.mean(smoothed)) / (MAX_THICKNESS * 2.0)
        std_t  = float(np.std(smoothed))  / (MAX_THICKNESS * 2.0)
        grms   = float(np.sqrt(np.mean(grad_mag**2))) / (UNEVEN_GRADIENT_THRESH * 2.0)

        obs = np.array([unpainted, weak, good, over, uneven,
                        np.clip(mean_t, 0, 1),
                        np.clip(std_t,  0, 1),
                        np.clip(grms,   0, 1)], dtype=np.float32)
        return obs

    # ─────────────────────────────────────────────────────────
    #  REWARD
    # ─────────────────────────────────────────────────────────

    def _compute_reward(self, obs: np.ndarray, ik_penalty: float = 0.0) -> float:
        unpainted, weak, good, over, uneven = obs[0], obs[1], obs[2], obs[3], obs[4]
        reward = (good * 200.0 - unpainted * 80.0 - over * 60.0
                  - uneven * 40.0 - weak * 20.0
                  - ik_penalty)
        return float(reward)

    # ─────────────────────────────────────────────────────────
    #  ENSEMBLE ACTION SELECTION
    # ─────────────────────────────────────────────────────────

    def _select_action(self, obs: np.ndarray):
        """
        Blend PPO (stochastic) and TD3 (deterministic) actions.
        TD3 weight grows from 0 → TD3_MAX_BLEND as buffer fills past warmup.
        Returns (blended_raw_action, ppo_log_prob, ppo_value).
        """
        ppo_raw, ppo_lp, ppo_val = self.ppo.select_action(obs)

        buf_len   = len(self.td3.buffer)
        td3_weight = min(TD3_MAX_BLEND,
                         TD3_MAX_BLEND * max(0, buf_len - TD3_WARMUP_STEPS)
                         / max(1, TD3_WARMUP_STEPS))

        if td3_weight > 0.0 and buf_len >= TD3_BATCH_SIZE:
            # TD3 uses exploration noise only during warmup
            noise  = 0.1 if buf_len < TD3_WARMUP_STEPS * 2 else 0.0
            td3_raw = self.td3.select_action(obs, noise=noise)
            blended = (1.0 - td3_weight) * ppo_raw + td3_weight * td3_raw
        else:
            blended = ppo_raw

        return blended, ppo_lp, ppo_val, td3_weight

    # ─────────────────────────────────────────────────────────
    #  DEFECT DETECTION  (kept for markers + path seeding)
    # ─────────────────────────────────────────────────────────

    def _detect_defects(self, smoothed: np.ndarray):
        gy, gx  = np.gradient(smoothed)
        grad_mag = np.sqrt(gx**2 + gy**2)

        unpainted_mask = smoothed < 1.0
        weak_mask      = (smoothed >= 1.0) & (smoothed < MIN_THICKNESS)
        over_mask      = smoothed > MAX_THICKNESS
        uneven_mask    = (grad_mag > UNEVEN_GRADIENT_THRESH) & ~unpainted_mask

        defect_cells = {
            0: np.argwhere(unpainted_mask).tolist(),
            1: np.argwhere(weak_mask).tolist(),
            3: np.argwhere(over_mask).tolist(),
            4: np.argwhere(uneven_mask).tolist(),
        }
        return defect_cells

    def _find_all_unpainted_clusters(self, smoothed: np.ndarray) -> list:
        """
        Return ALL clusters of unpainted/weak cells sorted by size (largest
        first).  Used so the path covers every missed region, not just the
        largest centroid.
        """
        unpainted_mask = (smoothed < MIN_THICKNESS)
        cells = np.argwhere(unpainted_mask).tolist()
        return self._cluster_defect_cells(cells)

    def _cluster_defect_cells(self, cells: list) -> list:
        if not cells:
            return []
        arr = np.array(cells, dtype=np.int32)
        if not _HAVE_SKLEARN or len(arr) < DBSCAN_MIN_SAMPLES:
            return [arr]
        labels = DBSCAN(eps=DBSCAN_EPS,
                        min_samples=DBSCAN_MIN_SAMPLES).fit_predict(arr)
        regions = [arr[labels == l] for l in np.unique(labels) if l >= 0]
        regions.sort(key=len, reverse=True)
        return regions

    # ─────────────────────────────────────────────────────────
    #  PATH GENERATION  (RL-directed boustrophedon patch)
    # ─────────────────────────────────────────────────────────

    def _surface_normal_from_ee(self):
        with self._tracking_lock:
            ee_pose = self._current_ee_pose

        if ee_pose is None:
            return DEFAULT_SURFACE_NX, DEFAULT_SURFACE_NY, DEFAULT_SURFACE_NZ

        ox = ee_pose.orientation.x
        oy = ee_pose.orientation.y
        oz = ee_pose.orientation.z
        ow = ee_pose.orientation.w
        # Third column of rotation matrix = tool-Z = surface normal
        nx = 2.0 * (ox * oz + ow * oy)
        ny = 2.0 * (oy * oz - ow * ox)
        nz = 1.0 - 2.0 * (ox * ox + oy * oy)
        mag = math.sqrt(nx*nx + ny*ny + nz*nz)
        if mag > 1e-6:
            return nx / mag, ny / mag, nz / mag
        return DEFAULT_SURFACE_NX, DEFAULT_SURFACE_NY, DEFAULT_SURFACE_NZ

    def _patch_poses(self, target_y: float, target_z: float,
                     standoff: float,
                     nx: float, ny: float, nz: float,
                     stamp) -> list:
        with self._tracking_lock:
            ee_pose = self._current_ee_pose

        # Recover surface X from EE pose; fall back to standoff * nx for flat +X wall
        surface_x = (ee_pose.position.x + standoff * nx) if ee_pose is not None \
                    else (standoff * nx)

        qx, qy, qz, qw = _orientation_facing_normal(nx, ny, nz)

        y_start = np.clip(target_y - PATCH_HALF_WIDTH,  self.y_min, self.y_max)
        y_end   = np.clip(target_y + PATCH_HALF_WIDTH,  self.y_min, self.y_max)
        z_start = np.clip(target_z - PATCH_HALF_HEIGHT, self.z_min, self.z_max)
        z_end   = np.clip(target_z + PATCH_HALF_HEIGHT, self.z_min, self.z_max)

        poses = []
        z_vals = np.arange(z_start, z_end + 1e-6, PATCH_STEP)
        left_to_right = True

        for z in z_vals:
            y_vals = np.arange(y_start, y_end + 1e-6, PATCH_STEP)
            if not left_to_right:
                y_vals = y_vals[::-1]
            left_to_right = not left_to_right

            for y in y_vals:
                # Same formula as pose_from_surface() in cartesian_trajectory_controller.cpp
                nozzle_x = surface_x   - standoff * nx
                nozzle_y = float(y)    - standoff * ny
                nozzle_z = float(z)    - standoff * nz

                ps = PoseStamped()
                ps.header.frame_id = 'world'
                ps.header.stamp    = stamp
                ps.pose.position.x = nozzle_x
                ps.pose.position.y = nozzle_y
                ps.pose.position.z = nozzle_z
                ps.pose.orientation.x = qx
                ps.pose.orientation.y = qy
                ps.pose.orientation.z = qz
                ps.pose.orientation.w = qw
                poses.append(ps)

        return poses

    def _generate_paint_path(self, target_y: float, target_z: float,
                              standoff: float,
                              clusters: list | None = None) -> Path:
        """
        Build a boustrophedon raster Path for the RL correction pass.

        When `clusters` is provided (list of np.ndarray of [row, col] cells),
        one patch is generated per cluster centroid so every defect region is
        covered — not just the RL-blended target centroid.
        When omitted, a single patch centred on (target_y, target_z) is used.

        Surface normal is resolved from /spray/tracking_pose so the nozzle
        offset direction matches cartesian_trajectory_controller.cpp's pose_from_surface() exactly.
        Falls back to DEFAULT_SURFACE_N* when no tracking pose is available.
        """
        path = Path()
        path.header.frame_id = 'world'
        path.header.stamp    = self.get_clock().now().to_msg()
        stamp = path.header.stamp

        nx, ny, nz = self._surface_normal_from_ee()

        if clusters:
            # One boustrophedon patch per cluster centroid
            for cluster_arr in clusters:
                centroid = cluster_arr.mean(axis=0)   # [row, col]
                cy = self.y_min + centroid[1] * self.res
                cz = self.z_min + centroid[0] * self.res
                path.poses.extend(
                    self._patch_poses(cy, cz, standoff, nx, ny, nz, stamp))
        else:
            path.poses.extend(
                self._patch_poses(target_y, target_z, standoff, nx, ny, nz, stamp))

        return path

    # ─────────────────────────────────────────────────────────
    #  SPRAY GATE
    # ─────────────────────────────────────────────────────────

    def _set_spray(self, state: bool):
        msg = Bool(); msg.data = state
        self.enable_pub.publish(msg)

    # ─────────────────────────────────────────────────────────
    #  MAIN DECISION STEP
    # ─────────────────────────────────────────────────────────

    def _decision_step(self):
        self._episode_step += 1

        # ── Inspect ──────────────────────────────────────────────────────
        with self.lock:
            raw = self.thickness_grid.copy()
        smoothed = gaussian_filter(raw, sigma=1.0)

        # ── Observation ──────────────────────────────────────────────────
        obs = self._build_obs(smoothed)

        # ── Reward for previous transition ───────────────────────────────
        self._ik_penalty       = self._consume_ik_penalty()
        self._planning_penalty = self._consume_planning_penalty()
        reward = self._compute_reward(
            obs, ik_penalty=self._ik_penalty + self._planning_penalty)
        self._total_reward += reward

        # ── TD3: store transition from last step ──────────────────────────
        if self._prev_obs is not None and self._prev_action is not None:
            self.td3.buffer.add(
                self._prev_obs, self._prev_action,
                reward, obs, done=False)
            cl, al = self.td3.update()
            if cl is not None:
                self.get_logger().debug(
                    f'TD3 update: critic_loss={cl:.4f} '
                    f'actor_loss={al if al else "delayed"}'
                )

        # ── PPO: store transition ─────────────────────────────────────────
        if self._prev_obs is not None and self._prev_action is not None:
            _, prev_lp, prev_val = self.ppo.select_action(self._prev_obs)
            self.ppo.store(self._prev_obs, self._prev_action,
                           reward, prev_val, prev_lp, done=False)
            if self.ppo.ready():
                loss = self.ppo.update(obs, last_done=False)
                self.get_logger().debug(f'PPO update: loss={loss:.4f}')

        # ── Guard: silent during PASS 1 ───────────────────────────────────
        if not self._pass1_done:
            self.get_logger().info(
                f'[{self._episode_step}] PASS 1 in progress | '
                f'reward={reward:.2f} | monitoring only.',
                throttle_duration_sec=5.0)
            self._prev_obs    = obs
            self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
            return

        # ── Action selection (PPO + TD3 ensemble) ────────────────────────
        blended, ppo_lp, ppo_val, td3_w = self._select_action(obs)
        params = decode_action(blended, self.y_min, self.y_max,
                               self.z_min, self.z_max)

        standoff  = params['standoff']
        flow      = params['flow']
        target_y  = params['target_y']
        target_z  = params['target_z']

        # Enforce the actual reachable workspace bounds before building the path.
        standoff = float(np.clip(standoff, STANDOFF_MIN, STANDOFF_MAX))
        target_y = float(np.clip(target_y, self.y_min, self.y_max))
        target_z = float(np.clip(target_z, self.z_min, self.z_max))

        # ── Spray OFF while travelling ────────────────────────────────────
        self._set_spray(False)

        # ── Defect check ─────────────────────────────────────────────────
        defect_cells = self._detect_defects(smoothed)
        any_defects  = any(len(v) > 0 for v in defect_cells.values())

        if any_defects:
            # ── Detect all unpainted clusters so every missed region is
            #    covered, not just the largest one ─────────────────────────
            priority_cells = (defect_cells.get(0, []) or   # unpainted first
                              defect_cells.get(1, []) or   # then weak
                              defect_cells.get(3, []) or   # then over
                              defect_cells.get(4, []))     # then uneven
            regions = self._cluster_defect_cells(priority_cells)

            if regions:
                # Bias RL target toward the largest cluster centroid
                cluster_centroid = regions[0].mean(axis=0)   # [row, col]
                c_y = self.y_min + cluster_centroid[1] * self.res
                c_z = self.z_min + cluster_centroid[0] * self.res
                # Soft blend: 70% cluster, 30% RL choice
                target_y = 0.7 * c_y + 0.3 * target_y
                target_z = 0.7 * c_z + 0.3 * target_z

            # Build a path that covers ALL unpainted clusters (multi-patch
            # boustrophedon).  Falls back to single centred patch when no
            # clusters were found (edge case with DBSCAN off / tiny regions).
            all_clusters = self._find_all_unpainted_clusters(smoothed)
            if all_clusters:
                paint_path = self._generate_paint_path(
                    target_y, target_z, standoff, clusters=all_clusters)
            else:
                paint_path = self._generate_paint_path(
                    target_y, target_z, standoff)

            n_poses = len(paint_path.poses)

            if n_poses > 0:
                # Spray ON gate: only enable spray when we are about to move
                # (cartesian_trajectory_controller.cpp gate controls the actual actuator signal;
                # this enable tells it we WANT spray on during this path).
                self._set_spray(True)
                self.path_pub.publish(paint_path)
                self.get_logger().info(
                    f'[{self._episode_step}] Path: {n_poses} wp | '
                    f'{len(all_clusters)} cluster(s) | '
                    f'standoff={standoff:.3f}m flow={flow:.2f} | '
                    f'target=({target_y:.3f},{target_z:.3f}) | '
                    f'td3_blend={td3_w:.2f} | reward={reward:.2f}')
            else:
                self.get_logger().info('Path empty after clipping — skipping.')
        else:
            # No defects: spray OFF, publish empty path to cancel any pending
            self._set_spray(False)
            self.path_pub.publish(Path())
            self.get_logger().info(
                f'[{self._episode_step}] No defects | reward={reward:.2f} | '
                f'spray OFF.')

        self.get_logger().info(
            f'STOP-CHECK any_defects={any_defects} '
            f'target=({target_y:.3f},{target_z:.3f}) '
            f'ik_penalty={self._ik_penalty:.2f} '
            f'planning_penalty={self._planning_penalty:.2f} '
            f'log_std={np.round(self.ppo.log_std, 3).tolist()}'
        )

        # ── Publish RL action (spray parameters for spray_sim_node) ──────
        act_msg = Float32MultiArray()
        act_msg.data = [standoff, flow]
        self.action_pub.publish(act_msg)

        # ── Reward ───────────────────────────────────────────────────────
        r_msg = Float32(); r_msg.data = reward
        self.reward_pub.publish(r_msg)

        # ── Status ───────────────────────────────────────────────────────
        quality_pct = obs[2] * 100.0
        status_str = (
            f'[Step {self._episode_step}] '
            f'Quality={quality_pct:.1f}% | '
            f'standoff={standoff:.3f}m flow={flow:.2f} | '
            f'td3_blend={td3_w:.2f} | '
            f'buf={len(self.td3.buffer)}/{TD3_WARMUP_STEPS} | '
            f'reward={reward:.2f} totalR={self._total_reward:.1f} | '
            f'unpainted={obs[0]*100:.1f}% weak={obs[1]*100:.1f}% '
            f'over={obs[3]*100:.1f}% uneven={obs[4]*100:.1f}%'
        )
        s_msg = String(); s_msg.data = status_str
        self.status_pub.publish(s_msg)
        self.get_logger().info(status_str)

        # ── RViz defect markers ───────────────────────────────────────────
        self._publish_defect_markers(defect_cells)

        # ── Periodic checkpoint ───────────────────────────────────────────
        if self._episode_step % 20 == 0:
            self._save_checkpoints()

        # ── Store for next step ───────────────────────────────────────────
        self._prev_obs    = obs
        self._prev_action = blended

    # ─────────────────────────────────────────────────────────
    #  RVIZ MARKERS
    # ─────────────────────────────────────────────────────────

    def _publish_defect_markers(self, defect_cells: dict):
        ma = MarkerArray()
        colours = {0:(1.,1.,1.,.8), 1:(1.,.5,0.,.8),
                   3:(1.,0.,0.,.8), 4:(.8,0.,1.,.8)}

        # Resolve surface X from latest EE pose (EE.x + DEFAULT_STANDOFF * nx)
        with self._tracking_lock:
            ee_pose = self._current_ee_pose
        nx, _, _ = self._surface_normal_from_ee()
        default_standoff = (STANDOFF_MIN + STANDOFF_MAX) / 2.0
        surface_x = (ee_pose.position.x + default_standoff * nx) \
                    if ee_pose is not None else default_standoff * nx
        marker_id = 0
        for dtype, cells in defect_cells.items():
            if not cells:
                continue
            r, g, b, a = colours.get(dtype, (.5,.5,.5,.5))
            sampled = cells[::max(1, len(cells) // 200)]
            for row, col in sampled:
                m = Marker()
                m.header.frame_id = 'world'
                m.ns = f'defect_{dtype}'; m.id = marker_id
                m.type = Marker.SPHERE; m.action = Marker.ADD
                m.pose.position.x = surface_x
                m.pose.position.y = self.y_min + col * self.res
                m.pose.position.z = self.z_min + row * self.res
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.025
                m.color.r = r; m.color.g = g
                m.color.b = b; m.color.a = a
                m.lifetime.sec = int(DECISION_INTERVAL * 2)
                ma.markers.append(m); marker_id += 1
        self.marker_pub.publish(ma)

    # ─────────────────────────────────────────────────────────
    #  CLEANUP
    # ─────────────────────────────────────────────────────────

    def destroy_node(self):
        self.get_logger().info(
            f'Shutting down | steps={self._episode_step} | '
            f'total_reward={self._total_reward:.2f}')
        self._set_spray(False)
        self._save_checkpoints()
        super().destroy_node()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = RLAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()