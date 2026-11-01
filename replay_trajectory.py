#!/usr/bin/env python3
import argparse
import csv
import math
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

ROOT = Path('/workspace/mobile_crane/recordings')
ARM = [f'crane_plus_joint{i}' for i in range(1, 5)]
HAND = ['crane_plus_joint_hand']
HOME = [0.0, -1.16, -2.01, -0.73, 0.0]


class Replay(Node):
    def __init__(self):
        super().__init__('recorded_trajectory_replay')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.current = {}

        self.create_subscription(
            JointState, '/joint_states', self.on_state, 10
        )

        self.arm = ActionClient(
            self, FollowJointTrajectory,
            '/crane_plus_arm_controller/follow_joint_trajectory'
        )
        self.hand = ActionClient(
            self, FollowJointTrajectory,
            '/crane_plus_gripper_controller/follow_joint_trajectory'
        )

    def on_state(self, msg):
        self.current.update(dict(zip(msg.name, msg.position)))

    def read_state(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(n in self.current for n in ARM + HAND):
                return [self.current[n] for n in ARM + HAND]
        raise RuntimeError('関節状態を取得できません')

    def execute(self, rows):
        if not self.arm.wait_for_server(timeout_sec=5):
            raise RuntimeError('アームに接続できません')
        if not self.hand.wait_for_server(timeout_sec=5):
            raise RuntimeError('グリッパに接続できません')

        # 両方の軌道に共通の開始時刻を設定
        start_ns = (
            self.get_clock().now().nanoseconds
            + 3_000_000_000
        )
        if start_ns <= 3_000_000_000:
            raise RuntimeError('/clockを確認してください')

        futures = []

        for client, indices, names in [
            (self.arm, range(4), ARM),
            (self.hand, [4], HAND),
        ]:
            goal = FollowJointTrajectory.Goal()
            traj = goal.trajectory
            traj.joint_names = names

            traj.header.stamp.sec = start_ns // 1_000_000_000
            traj.header.stamp.nanosec = start_ns % 1_000_000_000

            for t, angles in rows:
                point = JointTrajectoryPoint()
                point.positions = [angles[i] for i in indices]

                ns = round((t + 0.2) * 1_000_000_000)
                point.time_from_start.sec = ns // 1_000_000_000
                point.time_from_start.nanosec = ns % 1_000_000_000

                traj.points.append(point)

            futures.append(client.send_goal_async(goal))

        handles = []
        for future in futures:
            rclpy.spin_until_future_complete(self, future)
            handle = future.result()

            if handle is None or not handle.accepted:
                raise RuntimeError('軌道が拒否されました')
            handles.append(handle)

        print('アーム・グリッパ両方の軌道が受理されました')

        results = [h.get_result_async() for h in handles]

        for name, future in zip(['arm', 'gripper'], results):
            rclpy.spin_until_future_complete(self, future)
            result = future.result().result
            print(
                f'{name}: error_code={result.error_code} '
                f'{result.error_string}'
            )
            if result.error_code != 0:
                raise RuntimeError(f'{name}の再生に失敗しました')


def load_csv(path):
    names = ARM + HAND

    with open(path, newline='') as f:
        data = list(csv.DictReader(f))

    if not data:
        raise RuntimeError(f'CSVが空です: {path}')

    rows = [
        (
            float(r['relative_time_sec']),
            [float(r[n]) for n in names]
        )
        for r in data
    ]

    if any(
        not math.isfinite(v)
        for t, a in rows
        for v in [t] + a
    ):
        raise RuntimeError('CSVに無効な数値があります')

    if any(
        b[0] <= a[0]
        for a, b in zip(rows, rows[1:])
    ):
        raise RuntimeError('時刻の順序が不正です')

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        choices=['inspect', 'home', 'pick', 'place']
    )
    args = parser.parse_args()

    rows = None

    if args.mode in ('pick', 'place'):
        rows = load_csv(
            ROOT / f'{args.mode}_trajectory.csv'
        )

    if args.mode == 'inspect':
        for mode in ('pick', 'place'):
            x = load_csv(ROOT / f'{mode}_trajectory.csv')
            print(
                f'{mode}: {len(x)}点 / '
                f'{x[-1][0]:.3f}秒'
            )
            print(
                ' 開始:',
                [round(v, 4) for v in x[0][1]]
            )
            print(
                ' 終了:',
                [round(v, 4) for v in x[-1][1]]
            )
        return

    rclpy.init()
    node = Replay()

    try:
        actual = node.read_state()

        desired = (
            HOME if args.mode == 'home'
            else rows[0][1]
        )

        error = max(
            abs(a - b)
            for a, b in zip(actual, desired)
        )

        print(
            '現在姿勢:',
            [round(x, 3) for x in actual]
        )
        print(
            '開始姿勢との差:',
            round(error, 3), 'rad'
        )

        if args.mode != 'home' and error > 0.15:
            raise RuntimeError(
                '開始姿勢が合いません。'
                '前段階の動作を確認してください'
            )

        if args.mode == 'home':
            rows = [(3.0, HOME)]

        if args.mode == 'pick':
            corrected = []
            for t, angles in rows:
                angles = angles.copy()

                # 把持完了付近から、元のC++の閉じる目標値を維持
                if t >= 9.8:
                    angles[4] = math.radians(10.0)

                # 全動作を1.5倍の時間でゆっくり再生
                corrected.append((t * 1.5, angles))

            rows = corrected

        if args.mode == 'place':
            # 元の記録からグリッパが開き始める時刻を検出
            opening_times = [
                t for t, a in rows
                if t >= 3.0 and a[4] < 0.06
            ]

            if not opening_times:
                raise RuntimeError(
                    'グリッパの解放区間が見つかりません'
                )

            release_time = opening_times[0]
            corrected = []

            for t, angles in rows:
                angles = angles.copy()

                # 解放開始までは元のC++の閉じる目標を維持
                if t < release_time:
                    angles[4] = math.radians(10.0)

                # ゆっくり再生
                corrected.append((t * 1.5, angles))

            rows = corrected
            print(
                f'解放区間の開始目安: '
                f'{release_time:.3f}秒'
            )

        node.execute(rows)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
