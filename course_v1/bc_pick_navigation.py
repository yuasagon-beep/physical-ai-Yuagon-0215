#!/usr/bin/env python3
"""Small behavior-cloning demo for the mobile CRANE+ V2.

Train: numpy-only imitation of a pose-feedback driving expert in a fast 2D model.
Deploy: run the trained network (NOT the expert) on Gazebo via ROS2 cmd_vel.
This learns navigation to the Pick stop, not grasping or load retention.
"""
import argparse
import csv
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np

TARGET = (0.0, 0.0, 0.0)
TOPIC = '/model/mobile_crane/cmd_vel'
DEFAULT_MODEL = Path('/workspace/mobile_crane/course_v1/bc_pick_policy.npz')


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def features(x, y, yaw, target=TARGET):
    dx, dy = target[0] - x, target[1] - y
    rho = math.hypot(dx, dy)
    alpha = wrap(math.atan2(dy, dx) - yaw)
    beta = wrap(target[2] - yaw - alpha)
    return np.array([rho, math.sin(alpha), math.cos(alpha),
                     math.sin(beta), math.cos(beta)], dtype=np.float64)


def expert(obs):
    """Polar-coordinate pose controller (only for generating teacher labels)."""
    rho, sin_alpha, cos_alpha, sin_beta, cos_beta = obs
    alpha = math.atan2(sin_alpha, cos_alpha)
    beta = math.atan2(sin_beta, cos_beta)
    v = min(0.10, 0.8 * rho) * max(0.0, math.cos(alpha))
    w = float(np.clip(4.0 * alpha - 1.5 * beta, -0.48, 0.48))
    return np.array([v, w], dtype=np.float64)


def make_data(seed=2026):
    rng = np.random.default_rng(seed)
    observations = []
    labels = []

    # Expert demonstrations from different starts: true sequential episodes.
    for _ in range(180):
        x = rng.uniform(-0.75, -0.30)
        y = rng.uniform(-0.12, 0.12)
        yaw = rng.uniform(-0.25, 0.25)
        for _ in range(125):
            obs = features(x, y, yaw)
            vel = expert(obs)
            observations.append(obs)
            labels.append(vel)
            if abs(x) < 0.007 and abs(y) < 0.009 and abs(wrap(yaw)) < 0.025:
                break
            dt = 0.40
            x += float(vel[0]) * math.cos(yaw) * dt
            y += float(vel[0]) * math.sin(yaw) * dt
            yaw = wrap(yaw + float(vel[1]) * dt)

    # Off-path correction examples to avoid training only a memorized straight line.
    for _ in range(13000):
        x = rng.uniform(-0.80, 0.11)
        y = rng.uniform(-0.20, 0.20)
        yaw = rng.uniform(-0.55, 0.55)
        obs = features(x, y, yaw)
        observations.append(obs)
        labels.append(expert(obs))

    # Lots of near-goal examples teach slowing down and stopping.
    for _ in range(6000):
        x = rng.uniform(-0.075, 0.075)
        y = rng.uniform(-0.075, 0.075)
        yaw = rng.uniform(-0.18, 0.18)
        obs = features(x, y, yaw)
        observations.append(obs)
        labels.append(expert(obs))

    xs = np.asarray(observations, dtype=np.float64)
    ys = np.asarray(labels, dtype=np.float64)
    return xs, ys


def init_params(rng, dims=(5, 64, 64, 2)):
    params = {}
    for i in range(len(dims) - 1):
        params[f'W{i}'] = rng.normal(0, math.sqrt(2.0 / dims[i]), (dims[i], dims[i+1]))
        params[f'b{i}'] = np.zeros(dims[i+1], dtype=np.float64)
    return params


def predict(features_batch, params):
    a = np.asarray(features_batch, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    for i in range(3):
        a = a @ params[f'W{i}'] + params[f'b{i}']
        if i < 2:
            a = np.tanh(a)
    return a


def train(path=DEFAULT_MODEL):
    x, y = make_data()
    rng = np.random.default_rng(117)
    order = rng.permutation(len(x))
    cut = int(len(x) * 0.90)
    train_ids, val_ids = order[:cut], order[cut:]
    # Fixed scales; no train/real distribution mismatch.
    x = x / np.array([1.0, 1, 1, 1, 1])
    y = y / np.array([0.10, 0.48])
    params = init_params(rng)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(val) for k, val in params.items()}
    step = 0
    epochs = 32
    for epoch in range(epochs):
        rng.shuffle(train_ids)
        for ids in np.array_split(train_ids, max(1, len(train_ids)//256)):
            a0 = x[ids]
            z1 = a0 @ params['W0'] + params['b0']
            a1 = np.tanh(z1)
            z2 = a1 @ params['W1'] + params['b1']
            a2 = np.tanh(z2)
            pred = a2 @ params['W2'] + params['b2']
            # Mean squared error; gradient over batch and output dims.
            grad = (pred - y[ids]) / len(ids)
            grads = {
                'W2': a2.T @ grad,
                'b2': grad.sum(axis=0),
            }
            da2 = (grad @ params['W2'].T) * (1 - a2**2)
            grads['W1'] = a1.T @ da2
            grads['b1'] = da2.sum(axis=0)
            da1 = (da2 @ params['W1'].T) * (1 - a1**2)
            grads['W0'] = a0.T @ da1
            grads['b0'] = da1.sum(axis=0)
            step += 1
            lr = 0.002
            for key, g in grads.items():
                m[key] = 0.9 * m[key] + 0.1 * g
                v[key] = 0.999 * v[key] + 0.001 * (g*g)
                mh = m[key] / (1 - 0.9**step)
                vh = v[key] / (1 - 0.999**step)
                params[key] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if epoch in (0, 5, 11, 19, 31):
            err = predict(x[val_ids], params) - y[val_ids]
            rmse = float(np.sqrt(np.mean(err**2)))
            print(f'[TRAIN] epoch={epoch+1:02d}/{epochs}, validation RMSE(scaled)={rmse:.4f}', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **params)
    print(f'[SAVED] {path} | demonstrations+augmented states: {len(x)}')
    return params


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def policy(state, params):
    obs = features(*state) / np.array([1.0, 1, 1, 1, 1])
    action = predict(obs, params)[0] * np.array([0.10, 0.48])
    return float(np.clip(action[0], -0.10, 0.10)), float(np.clip(action[1], -0.48, 0.48))


def reached(x, y, yaw):
    return abs(x) < .012 and abs(y) < .018 and abs(wrap(yaw)) < .05


def evaluate(params, n=60):
    rng = np.random.default_rng(1312)
    good = 0
    stats = []
    for _ in range(n):
        x = rng.uniform(-0.68, -0.37)
        y = rng.uniform(-0.06, 0.06)
        yaw = rng.uniform(-0.18, 0.18)
        for t in range(170):
            if reached(x, y, yaw):
                good += 1
                stats.append(t)
                break
            v, w = policy((x, y, yaw), params)
            x += v * math.cos(yaw) * 0.4
            y += v * math.sin(yaw) * 0.4
            yaw = wrap(yaw + w * 0.4)
    print(f'[2D SIM EVAL] reached Pick goal: {good}/{n} ({good/n:.1%})', flush=True)
    if stats:
        print(f'[2D SIM EVAL] average steps among successes: {sum(stats)/len(stats):.1f}', flush=True)
    print('NOTE: 2D evaluation is NOT proof of success in Gazebo.')


POSE_PATTERN = re.compile(r'\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]')

def gazebo_pose(name='mobile_crane'):
    last = ''
    for retry in range(3):
        try:
            proc = subprocess.run(['ign', 'model', '-m', name, '--pose'], capture_output=True, text=True, timeout=10)
            last = proc.stdout + proc.stderr
            matches = POSE_PATTERN.findall(last)
            if proc.returncode == 0 and len(matches) >= 2:
                xyz = [float(t) for t in matches[0]]
                rpy = [float(t) for t in matches[1]]
                return xyz, rpy
        except subprocess.TimeoutExpired:
            last = 'Gazebo state query timed out'
        print(f'[GAZEBO] position query retry {retry+1}/3', flush=True)
        time.sleep(0.4)
    raise RuntimeError(f'Cannot read {name} pose: {last[-300:]}')


def run_gazebo(params, log_path):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    rclpy.init()
    node = Node('bc_pick_navigation')
    pub = node.create_publisher(Twist, TOPIC, 10)
    def stop():
        for _ in range(6):
            pub.publish(Twist())
            time.sleep(.05)
    try:
        if node.count_subscribers(TOPIC) == 0:
            raise RuntimeError('cmd_vel bridge not connected. Start ros_gz_bridge first.')
        xyz, rpy = gazebo_pose()
        print(f'[START] x={xyz[0]:.3f} y={xyz[1]:.3f} yaw={rpy[2]:.3f}')
        if not (-0.85 <= xyz[0] <= -0.20 and abs(xyz[1]) <= .20 and abs(rpy[2]) <= .50):
            raise RuntimeError('Robot is not near the start corridor. Reset course first.')
        cube, _ = gazebo_pose('product_cube')
        if not (0.14 < cube[2] < .20):
            raise RuntimeError('Box is not on its pickup stand. Reset first.')
        print('[AI] trained neural network controls BASE only. No scripted expert, no Pick command.')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step','x','y','yaw','linear_cmd','angular_cmd'])
            for step in range(85):
                xyz, rpy = gazebo_pose()
                x, y, yaw = xyz[0], xyz[1], rpy[2]
                if not (-0.88 < x < .05 and abs(y) < .24 and abs(yaw) < .9):
                    raise RuntimeError(f'Course safety limit reached at ({x:.3f},{y:.3f},{yaw:.3f})')
                if reached(x, y, yaw):
                    print(f'[SUCCESS] AI arrived at Pick goal, x={x:.4f} y={y:.4f} yaw={yaw:.4f}')
                    return True
                linear, angular = policy((x,y,yaw), params)
                if step % 5 == 0:
                    print(f'[AI] step {step:02d}: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}; v={linear:+.3f}, w={angular:+.3f}', flush=True)
                writer.writerow([step, x,y,yaw,linear,angular])
                f.flush()
                msg = Twist()
                msg.linear.x = linear
                msg.angular.z = angular
                until = time.monotonic() + .46
                while time.monotonic() < until:
                    pub.publish(msg)
                    time.sleep(.05)
                stop()
            raise RuntimeError('AI did not reach goal within 85 steps')
    finally:
        stop()
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('train','eval','gazebo'))
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--log', type=Path, default=Path('/workspace/mobile_crane/course_v1/experiment_logs/bc_gazebo.csv'))
    args = parser.parse_args()
    if args.mode == 'train':
        params = train(args.model)
        evaluate(params)
    elif args.mode == 'eval':
        evaluate(load(args.model))
    else:
        run_gazebo(load(args.model), args.log)

if __name__ == '__main__':
    main()
