#!/usr/bin/env python3
import math
import re
import subprocess
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

BASE = "/workspace/mobile_crane"

def pose(name):
    p = subprocess.run(
        ["ign", "model", "-m", name, "--pose"],
        capture_output=True, text=True, timeout=15
    )
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)

    matches = re.findall(
        r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]",
        p.stdout
    )
    if len(matches) < 2:
        raise RuntimeError("位置を取得できません")
    return (
        list(map(float, matches[0])),
        list(map(float, matches[1]))
    )

def command(pub, linear=0, angular=0, duration=.5):
    msg = Twist()
    msg.linear.x = float(linear)
    msg.angular.z = float(angular)

    stop = Twist()
    end = time.monotonic() + duration

    while time.monotonic() < end:
        pub.publish(msg)
        time.sleep(.05)

    for _ in range(5):
        pub.publish(stop)
        time.sleep(.05)

def main():
    robot, angles = pose("mobile_crane")
    cube, _ = pose("product_cube")

    print("===== 連続Pick・後退実験 =====", flush=True)
    print("初期車体:", robot, flush=True)
    print("初期箱:", cube, flush=True)

    if abs(robot[0] + .55) > .03 or abs(robot[1]) > .03:
        raise RuntimeError(
            "スタート地点ではありません。リセットしてください"
        )

    if abs(angles[2]) > .05:
        raise RuntimeError("初期方向がずれています")

    if abs(cube[2] - .17) > .025:
        raise RuntimeError("箱が商品台にありません")

    rclpy.init()
    node = Node("continuous_pick_test")
    pub = node.create_publisher(
        Twist, "/model/mobile_crane/cmd_vel", 10
    )

    try:
        print("\n[1] W×12で前進", flush=True)
        for i in range(12):
            command(pub, linear=.10)
            print(f"W {i+1}/12", flush=True)

        robot, angles = pose("mobile_crane")
        print("12回後:", robot, flush=True)

        # 最後の1cm程度を短い指令で補正
        error = -robot[0]
        if .004 < error < .018:
            duration = min(.15, error / .10)
            print(
                f"位置を約{error*100:.1f}cm補正",
                flush=True
            )
            command(
                pub, linear=.10,
                duration=duration
            )

        robot, angles = pose("mobile_crane")
        print("Pick位置:", robot, flush=True)

        if (
            abs(robot[0]) > .02
            or abs(robot[1]) > .02
            or abs(angles[2]) > .04
        ):
            raise RuntimeError(
                "Pick位置がずれています。安全のため中止"
            )

        print("\n[2] Pick開始", flush=True)
        subprocess.run(
            ["python3", BASE + "/replay_trajectory.py", "pick"],
            check=True, timeout=80
        )

        cube, _ = pose("product_cube")
        print("Pick直後の箱:", cube, flush=True)

        if cube[2] < .27:
            raise RuntimeError(
                "箱が十分に持ち上がっていません。後退中止"
            )

        print("\n[3] 直ちにS×10で後退", flush=True)
        for i in range(10):
            command(pub, linear=-.10)
            print(f"S {i+1}/10", flush=True)

        robot, _ = pose("mobile_crane")
        cube, _ = pose("product_cube")

        print("\n===== 実験結果 =====", flush=True)
        print("車体:", robot, flush=True)
        print("箱:", cube, flush=True)

        if cube[2] >= .25:
            print(
                "箱の高さは維持されています。"
                "画面でも把持を確認してください。",
                flush=True
            )
        else:
            print(
                "箱が滑った・落ちた可能性があります。",
                flush=True
            )

    finally:
        for _ in range(10):
            pub.publish(Twist())
            time.sleep(.05)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
