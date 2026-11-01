#!/usr/bin/env python3
"""Soft reset for course_v1 (manual optional; full Gazebo restart is cleaner).

Run inside the ROS2 Humble container with Gazebo, RSP, ros_gz_control up.
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

BASE=Path('/workspace/mobile_crane')
STATE=BASE/'course_v1/course_config.json'
WORLD='mobile_world'


def run(args, timeout=40):
    p=subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{p.stdout}{p.stderr}")
    return p.stdout+p.stderr


def pose(name):
    out=run(['ign','model','-m',name,'--pose'])
    vals=re.findall(r'\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]',out)
    if len(vals)<2:
        raise RuntimeError(f'Cannot parse pose of {name}: {out}')
    return tuple(map(float, vals[0])),tuple(map(float, vals[1]))


def set_pose(name,x,y,z,yaw=0):
    request=(f'name: "{name}" '
             f'position: {{x: {x:.7f}, y: {y:.7f}, z: {z:.7f}}} '
             f'orientation: {{x: 0, y: 0, z: {math.sin(yaw/2):.9f}, w: {math.cos(yaw/2):.9f}}}')
    response=run(['ign','service','-s',f'/world/{WORLD}/set_pose',
                  '--reqtype','ignition.msgs.Pose',
                  '--reptype','ignition.msgs.Boolean',
                  '--timeout','5000','--req',request])
    if not re.search(r'data:\s*true\b',response):
        raise RuntimeError(f'Set pose rejected for {name}: {response}')
    print('Set pose:',name,flush=True)


def stop():
    for i in range(3):
        run(['ign','topic','-t','/model/mobile_crane/cmd_vel',
             '-m','ignition.msgs.Twist',
             '-p','linear: {x: 0}, angular: {z: 0}'])
        time.sleep(.15)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true',help='Actually reset entities')
    args=parser.parse_args()
    config=json.loads(STATE.read_text())
    required=['mobile_crane','pickup_stand','product_cube','delivery_stand','ghost_pick','ghost_drop']
    listed=run(['ign','model','--list'])
    names=set(re.findall(r'^\s*-\s+(\S+)\s*$',listed,re.M))
    missing=set(required)-names
    if missing: raise RuntimeError(f'Course is not active; missing models: {missing}')
    print('===== Course V1 soft reset =====')
    for key in ('mobile_crane','product_cube'):
        print(key, pose(key))
    print('Target robot:',config['start'])
    print('Target cube:',config['product_cube'])
    if not args.apply:
        print('DRY RUN ONLY. Use --apply to execute the reset.')
        return
    print('Stopping base...',flush=True)
    stop()
    print('Moving cube to isolation spot...',flush=True)
    set_pose('product_cube', 5.0, 5.0, .25)
    st=config['start']
    print('Returning robot to start...',flush=True)
    set_pose('mobile_crane', st['x'],st['y'],0,st['yaw'])
    stop()
    print('Returning arm to Home...',flush=True)
    run([sys.executable,str(BASE/'replay_trajectory.py'),'home'],timeout=45)
    time.sleep(1)
    print('Resetting robot pose again to reduce teleport drift...',flush=True)
    set_pose('mobile_crane',st['x'],st['y'],0,st['yaw'])
    stop()
    cube=config['product_cube']
    print('Returning cube to pickup stand...',flush=True)
    set_pose('product_cube',cube['x'],cube['y'],cube['z'])
    time.sleep(1)
    print('Robot:',pose('mobile_crane'))
    print('Cube:',pose('product_cube'))
    print('Soft reset completed. Visually confirm box settles and robot is stationary.')
    print('NOTE: set_pose does not guarantee zero linear/angular velocities or a complete controller reset.')
    print('For reproducible training episodes, restart the simulation from course_world.sdf.')

if __name__=='__main__':
    main()
