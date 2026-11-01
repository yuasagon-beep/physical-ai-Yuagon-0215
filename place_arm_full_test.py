#!/usr/bin/env python3
import re
import subprocess
import rclpy

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from replay_trajectory import Replay, load_csv, ROOT, ARM

TIME_SCALE = 3.0

def cube_height():
    p = subprocess.run(
        ["ign", "model", "-m", "product_cube", "--pose"],
        capture_output=True, text=True, timeout=15
    )
    if p.returncode != 0:
        raise RuntimeError(p.stdout + p.stderr)

    matches = re.findall(
        r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]",
        p.stdout
    )
    if not matches:
        raise RuntimeError("箱の位置を取得できません")

    return float(matches[0][2])

def main():
    original = load_csv(ROOT / "place_trajectory.csv")

    opening = [
        t for t, a in original
        if t >= 3.0 and a[4] < 0.06
    ]
    if not opening:
        raise RuntimeError("解放開始時刻を検出できません")

    release_time = opening[0]
    selected = [
        (t, a) for t, a in original
        if t < release_time
    ]

    if not (3.5 < release_time < 4.5):
        raise RuntimeError("解放開始時刻が想定外です")

    z_before = cube_height()
    print("===== Place連続移動実験 =====")
    print(f"箱の開始高さ: {z_before:.4f} m")
    print(f"解放開始: CSV {release_time:.3f}秒")
    print(f"実行終了: CSV {selected[-1][0]:.3f}秒")
    print(f"実行時間: 約{selected[-1][0]*TIME_SCALE:.1f}秒")
    print("グリッパへの指令: なし")

    if z_before < 0.25:
        raise RuntimeError(
            "箱が把持高さにありません。実験を中止します"
        )

    rclpy.init()
    node = Replay()

    try:
        actual = node.read_state()
        desired = original[0][1][:4]

        error = max(
            abs(a - b)
            for a, b in zip(actual[:4], desired)
        )

        print(f"アーム開始姿勢誤差: {error:.4f} rad")
        print(f"グリッパ実測角度: {actual[4]:.4f} rad")

        if error > 0.10:
            raise RuntimeError(
                "Pick終了姿勢ではありません。中止します"
            )

        if not (0.04 <= actual[4] <= 0.19):
            raise RuntimeError(
                "グリッパの状態が想定外です"
            )

        if not node.arm.wait_for_server(timeout_sec=5):
            raise RuntimeError("アームに接続できません")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM

        now_ns = node.get_clock().now().nanoseconds
        if now_ns <= 0:
            raise RuntimeError("/clockを確認してください")

        start_ns = now_ns + 3_000_000_000
        goal.trajectory.header.stamp.sec = start_ns // 1_000_000_000
        goal.trajectory.header.stamp.nanosec = start_ns % 1_000_000_000

        for t, angles in selected:
            point = JointTrajectoryPoint()
            point.positions = angles[:4]

            ns = round((t * TIME_SCALE + 0.2) * 1_000_000_000)
            point.time_from_start.sec = ns // 1_000_000_000
            point.time_from_start.nanosec = ns % 1_000_000_000

            goal.trajectory.points.append(point)

        print("\nRelease直前まで連続移動を開始します")
        print("箱が滑るなど異常があればGazeboを一時停止してください")

        future = node.arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future)

        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("アーム軌道が拒否されました")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)

        result = result_future.result().result
        print("arm:", result.error_code, result.error_string)

        if result.error_code != 0:
            raise RuntimeError("アーム移動に失敗しました")

        z_after = cube_height()
        print(f"終了時の箱の高さ: {z_after:.4f} m")

        if z_after < 0.18:
            print("警告: 箱の落下が疑われます")
        else:
            print("軌道完了。Gazeboで把持状態を確認してください")

        print("グリッパの解放指令は送っていません")

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
