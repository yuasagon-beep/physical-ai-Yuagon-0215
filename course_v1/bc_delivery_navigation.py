#!/usr/bin/env python3
"""Behavior cloning for the remaining mobile CRANE+ V2 delivery route.

Stages: reverse from Pick -> turn left -> forward to Drop.
The controller is a trained small neural network, with explicit safety guardrails,
not a reinforcement-learning policy. The recorded arm Pick/Place trajectories are
optionally invoked in a full task; they are NOT learned by this model.

Intended location: /workspace/mobile_crane/course_v1/bc_delivery_navigation.py
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

COURSE = Path('/workspace/mobile_crane/course_v1')
BASE = COURSE.parent
MODEL = COURSE / 'bc_delivery_policy.npz'
LOGDIR = COURSE / 'experiment_logs'
TOPIC = '/model/mobile_crane/cmd_vel'

GOALS = {
    'reverse': (-0.55, 0.0, 0.0),
    'turn': (-0.55, 0.0, math.pi / 2),
    'forward': (-0.55, 1.0, math.pi / 2),
}
PHASES = ['reverse', 'turn', 'forward']
MODE = {'reverse': 0, 'turn': 1, 'forward': 2}
POSE_PATTERN = re.compile(r'\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]')


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def obs_from(x, y, yaw, phase):
    tx, ty, target_yaw = GOALS[phase]
    dx, dy = tx-x, ty-y
    along = math.cos(target_yaw)*dx + math.sin(target_yaw)*dy
    lateral = -math.sin(target_yaw)*dx + math.cos(target_yaw)*dy
    delta = wrap(target_yaw - yaw)
    phase_hot = [float(i == MODE[phase]) for i in range(3)]
    return np.asarray([along, lateral, math.sin(delta), math.cos(delta), *phase_hot], dtype=np.float64)


def expert(obs, phase):
    along, lateral, sy, cy = obs[:4]
    yaw_err = math.atan2(sy, cy)
    if phase == 'turn':
        return np.array([0., np.clip(1.7*yaw_err, -0.38, 0.38)])
    direction = -1. if phase == 'reverse' else 1.
    # A heading bias compensates lateral error, with opposite sign when reversing.
    bias = math.atan2(direction*1.2*lateral, max(abs(along), 0.19))
    bias = np.clip(bias, -0.20, 0.20)
    w = float(np.clip(2.3*wrap(bias + yaw_err), -0.38, 0.38))
    v = direction*min(0.10, 0.85*abs(along))
    v *= max(0.35, 1.0-0.8*abs(yaw_err))
    if abs(along) < 0.003:
        v = 0.
    return np.array([v, w])


def make_data(seed=2026):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    # Randomized single-step teacher samples, including close-goal and off-center states.
    for phase in PHASES:
        tx, ty, theta = GOALS[phase]
        if phase == 'reverse':
            bounds = (-.025, .06, -.58, .06, -.12, .12, -.18, .18)
        elif phase == 'turn':
            bounds = (-.59, -.51, -.045, .045, -.10, math.pi/2+.12)
        else:
            bounds = (-.63, -.47, -.07, 1.06, math.pi/2-.18, math.pi/2+.18)
        # More samples near target to teach fine stopping.
        for k in range(13500):
            if phase == 'reverse':
                x = rng.uniform(-.63, .08) if k < 9000 else rng.uniform(-.61, -.49)
                y = rng.uniform(-.075, .075) if k < 9000 else rng.uniform(-.027, .027)
                yaw = rng.uniform(-.18, .18) if k < 9000 else rng.uniform(-.045, .045)
            elif phase == 'turn':
                x = rng.uniform(-.61, -.49)
                y = rng.uniform(-.05, .05)
                yaw = rng.uniform(-.15, 1.77) if k < 9000 else rng.uniform(1.42, 1.72)
            else:
                x = rng.uniform(-.63, -.47) if k < 9000 else rng.uniform(-.574, -.526)
                y = rng.uniform(-.10, 1.08) if k < 9000 else rng.uniform(.90, 1.035)
                yaw = rng.uniform(1.39, 1.76) if k < 9000 else rng.uniform(1.52, 1.62)
            obs = obs_from(x,y,yaw,phase)
            xs.append(obs)
            ys.append(expert(obs, phase))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def init_params(rng, sizes=(7,48,48,2)):
    params = {}
    for i in range(len(sizes)-1):
        params[f'W{i}'] = rng.normal(0, math.sqrt(1.5/sizes[i]), (sizes[i],sizes[i+1]))
        params[f'b{i}'] = np.zeros(sizes[i+1])
    return params


def forward(x,p):
    a1 = np.tanh(x @ p['W0'] + p['b0'])
    a2 = np.tanh(a1 @ p['W1'] + p['b1'])
    out = a2 @ p['W2'] + p['b2']
    return a1, a2, out


def predict(x,p):
    x = np.asarray(x)
    if x.ndim == 1: x = x[None,:]
    return forward(x,p)[2]


def train(path=MODEL, epochs=32):
    x,y = make_data()
    scaled = y / np.array([.10, .38])
    rng = np.random.default_rng(117)
    idx = rng.permutation(len(x))
    train_idx, valid_idx = idx[:int(len(x)*.92)], idx[int(len(x)*.92):]
    p = init_params(rng)
    m = {k: np.zeros_like(t) for k,t in p.items()}
    v = {k: np.zeros_like(t) for k,t in p.items()}
    it = 0
    for e in range(epochs):
        rng.shuffle(train_idx)
        for inds in np.array_split(train_idx, max(1, len(train_idx)//256)):
            xx,yy = x[inds],scaled[inds]
            a1,a2,yh=forward(xx,p)
            gg=(yh-yy)/len(inds)
            grad={'W2':a2.T@gg,'b2':gg.sum(0)}
            da2=(gg@p['W2'].T)*(1-a2*a2)
            grad['W1']=a1.T@da2; grad['b1']=da2.sum(0)
            da1=(da2@p['W1'].T)*(1-a1*a1)
            grad['W0']=xx.T@da1; grad['b0']=da1.sum(0)
            it+=1
            for k,g in grad.items():
                m[k] = .9*m[k]+.1*g
                v[k] = .999*v[k]+.001*g*g
                p[k] -= .002*(m[k]/(1-.9**it))/(np.sqrt(v[k]/(1-.999**it))+1e-8)
        if (e+1) in (1,8,16,24,epochs):
            rmse = float(np.sqrt(np.mean((predict(x[valid_idx],p)-scaled[valid_idx])**2)))
            print(f'[TRAIN] epoch {e+1}/{epochs}, val RMSE={rmse:.4f}',flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**p)
    print(f'[SAVE] policy {path}, expert samples={len(x)}', flush=True)
    return p


def load(path=MODEL):
    with np.load(path) as f:
        return {k:f[k] for k in f.files}


def policy(x,y,yaw,phase,p):
    res = predict(obs_from(x,y,yaw,phase),p)[0]*np.array([.10,.38])
    if phase=='turn':
        return 0.,float(np.clip(res[1],-.4,.4))
    direction=-1 if phase=='reverse' else 1
    return direction*max(0.,float(np.clip(res[0]*direction,0.,.1))),float(np.clip(res[1],-.4,.4))


def reached(x,y,yaw,phase):
    gx,gy,ga=GOALS[phase]
    if phase=='turn':
        return math.hypot(x-gx,y-gy)<.040 and abs(wrap(ga-yaw))<.016
    if phase=='reverse':
        return math.hypot(x-gx,y-gy)<.018 and abs(wrap(ga-yaw))<.045
    return math.hypot(x-gx,y-gy)<.018 and abs(wrap(ga-yaw))<.028


def step2d(x,y,yaw,phase,p,dt=.40):
    v,w=policy(x,y,yaw,phase,p)
    return x+v*math.cos(yaw)*dt, y+v*math.sin(yaw)*dt,wrap(yaw+w*dt)


def evaluate(p,n=80):
    rng=np.random.default_rng(827)
    ok=0
    per_phase={ph:0 for ph in PHASES}
    failures=[]
    for i in range(n):
        x=rng.uniform(-.020,.012)
        y=rng.uniform(-.014,.014)
        yaw=rng.uniform(-.03,.03)
        stage_ok=True
        for phase in PHASES:
            for k in range({'reverse':85,'turn':55,'forward':110}[phase]):
                if reached(x,y,yaw,phase):
                    per_phase[phase]+=1
                    break
                x,y,yaw=step2d(x,y,yaw,phase,p)
            else:
                stage_ok=False
                failures.append((i,phase,round(x,4),round(y,4),round(yaw,3)))
                break
        if stage_ok:ok+=1
    print(f'[2D SIM] full Pick-to-Delivery route: {ok}/{n} ({ok/n:.1%})')
    print('[2D SIM] stage successes:',per_phase)
    if failures: print('[2D SIM] first failures:',failures[:5])
    print('NOTE: 2D SIM does NOT simulate grasping or prove Gazebo success.')
    return ok==n


def gazebo_pose(name):
    last=''
    for i in range(3):
        try:
            q=subprocess.run(['ign','model','-m',name,'--pose'],capture_output=True,text=True,timeout=12)
            last=q.stdout+q.stderr
            parts=POSE_PATTERN.findall(last)
            if q.returncode==0 and len(parts)>=2:
                return tuple(map(float,parts[0])),tuple(map(float,parts[1]))
        except subprocess.TimeoutExpired:
            last='pose query timed out'
        if i<2: time.sleep(.6)
    raise RuntimeError(f'Gazebo position unavailable for {name}: {last[-280:]}')


def check_running_and_ready():
    out=subprocess.run(['ros2','control','list_controllers','-c','/controller_manager'],capture_output=True,text=True,timeout=15)
    if out.returncode:
        raise RuntimeError('controller_manager not responding')
    clean=re.sub(r'\x1b\[[0-9;]*m','',out.stdout)
    for name in ('joint_state_broadcaster','crane_plus_arm_controller','crane_plus_gripper_controller'):
        if not any(line.split() and line.split()[0]==name and line.split()[-1]=='active' for line in clean.splitlines()):
            raise RuntimeError(f'{name} not active:\n{clean}')


def control(p, phases, do_pick=False, do_place=False, log=None):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    if log is None:
        LOGDIR.mkdir(parents=True,exist_ok=True)
        log=LOGDIR/f'bc_delivery_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    check_running_and_ready()
    rclpy.init()
    node=Node('bc_delivery_route')
    pub=node.create_publisher(Twist,TOPIC,10)
    def stop():
        for _ in range(7):
            pub.publish(Twist());time.sleep(.05)
    def command(v,w,duration=.4):
        msg=Twist();msg.linear.x=float(v);msg.angular.z=float(w)
        end=time.monotonic()+duration
        while time.monotonic()<end:
            pub.publish(msg);time.sleep(.05)
        stop()
    def grip_check(anchor):
        robot,rr=gazebo_pose('mobile_crane')
        cube,_=gazebo_pose('product_cube')
        dx=cube[0]-robot[0];dy=cube[1]-robot[1]
        y=rr[2]
        loc=(math.cos(y)*dx+math.sin(y)*dy,-math.sin(y)*dx+math.cos(y)*dy,cube[2]-robot[2])
        drift=math.hypot(loc[0]-anchor[0],loc[1]-anchor[1])
        dz=loc[2]-anchor[2]
        if cube[2]<.255 or drift>.055 or abs(dz)>.055:
            raise RuntimeError(f'Box dropped/slipping: cube_z={cube[2]:.3f}, drift_xy={drift:.3f}, drift_z={dz:.3f}')
    def call_script(cmd,timeout=120):
        proc=subprocess.run(cmd,timeout=timeout)
        if proc.returncode:raise RuntimeError('Command failed: '+' '.join(cmd))
    anchor=None
    try:
        if node.count_subscribers(TOPIC)==0:
            raise RuntimeError('cmd_vel ROS-Gazebo bridge is not connected')
        xyz,rpy=gazebo_pose('mobile_crane')
        if math.hypot(xyz[0],xyz[1])>.035 or abs(wrap(rpy[2]))>.05:
            raise RuntimeError('Robot is not at Pick ghost. Reset and run bc_pick_navigation.py gazebo first.')
        if do_pick:
            cube,_=gazebo_pose('product_cube')
            if not (.145<=cube[2]<=.20):
                raise RuntimeError('Box not on pickup stand, cannot Pick.')
            print('[PICK] replay known successful trajectory...',flush=True)
            call_script([sys.executable,str(BASE/'replay_trajectory.py'),'pick'],timeout=110)
            c,_=gazebo_pose('product_cube')
            if c[2]<.28:
                raise RuntimeError(f'Pick failed, cube_z={c[2]:.3f}')
            xyz,rpy=gazebo_pose('mobile_crane')
            dx,dy=c[0]-xyz[0],c[1]-xyz[1]
            anchor=(math.cos(rpy[2])*dx+math.sin(rpy[2])*dy,-math.sin(rpy[2])*dx+math.cos(rpy[2])*dy,c[2]-xyz[2])
            print('[PICK] lifted, relative anchor=',anchor,flush=True)
        print('[AI] BC neural network controls reverse, turning and forward phases.',flush=True)
        print('[AI] scripted stage transitions + safety checks; not PPO/RL.',flush=True)
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',newline='') as fh:
            writer=csv.writer(fh)
            writer.writerow(['phase','step','robot_x','robot_y','robot_yaw','linear_cmd','angular_cmd','cube_z'])
            for phase in phases:
                print(f'[AI] {phase} -> target {GOALS[phase]}',flush=True)
                max_steps={'reverse':85,'turn':60,'forward':130}[phase]
                for i in range(max_steps):
                    xyz,rpy=gazebo_pose('mobile_crane')
                    x,y,theta=xyz[0],xyz[1],rpy[2]
                    if phase=='reverse':
                        if not (-.69 <= x <= .12 and abs(y)<.12 and abs(wrap(theta))<.23):
                            raise RuntimeError(f'Reverse lane/safety boundary violated: {xyz}, yaw={theta:.3f}')
                    elif phase=='turn':
                        if math.hypot(x+.55,y)>.065:
                            raise RuntimeError('Turn must remain near start point')
                    else:
                        if not (-.68 <= x <= -.42 and -.09 <= y <= 1.08 and abs(wrap(theta-math.pi/2))<.24):
                            raise RuntimeError(f'Delivery lane/safety boundary violated: {xyz}, yaw={theta:.3f}')
                    if anchor is not None:
                        grip_check(anchor)
                    if reached(x,y,theta,phase):
                        stop()
                        print(f'[AI] {phase} reached: x={x:.4f},y={y:.4f},yaw={theta:.4f}',flush=True)
                        break
                    v,w=policy(x,y,theta,phase,p)
                    if i%5==0:
                        print(f'[AI] {phase} step={i:02d}: x={x:.3f},y={y:.3f}, yaw={math.degrees(theta):.1f}°, v={v:+.3f},w={w:+.3f}',flush=True)
                    cube,_=gazebo_pose('product_cube')
                    writer.writerow([phase,i,x,y,theta,v,w,cube[2]])
                    fh.flush()
                    # Existing bc_pick_navigation uses 0.46s pulses. This is shorter
                    # to limit overshoot near the precisely-placed delivery stand.
                    command(v,w,duration=.40)
                else:
                    raise RuntimeError(f'AI could not reach {phase} within {max_steps} steps')
        if do_place:
            if anchor is None:
                raise RuntimeError('Refusing Place without verified Pick')
            grip_check(anchor)
            print('[PLACE] arm-only trajectory ...',flush=True)
            call_script([sys.executable,str(BASE/'place_arm_full_test.py')],timeout=120)
            cube,_=gazebo_pose('product_cube')
            stand,_=gazebo_pose('delivery_stand')
            dist=math.hypot(cube[0]-stand[0],cube[1]-stand[1])
            if dist>.038 or not (.16<cube[2]<.32):
                raise RuntimeError(f'Refusing release: box not over stand, xy_error={dist:.3f}, z={cube[2]:.3f}')
            print('[RELEASE] open gripper...',flush=True)
            goal="{trajectory: {joint_names: ['crane_plus_joint_hand'], points: [{positions: [-0.5235], time_from_start: {sec: 2}}]}}"
            call_script(['ros2','action','send_goal','/crane_plus_gripper_controller/follow_joint_trajectory','control_msgs/action/FollowJointTrajectory',goal],timeout=45)
            time.sleep(3)
            cube,_=gazebo_pose('product_cube')
            xy=math.hypot(cube[0]-stand[0],cube[1]-stand[1])
            print('[RESULT] cube=',cube,'relative to stand=',xy,flush=True)
            if xy>.068 or not (.15<cube[2]<.24):
                raise RuntimeError('Release did not place box correctly')
            print('***** AI-NAVIGATION + BOX DELIVERY COMPLETE *****',flush=True)
        print('[AI] navigation complete. Logged:',log,flush=True)
    finally:
        stop()
        node.destroy_node()
        rclpy.shutdown()


def main():
    ap=argparse.ArgumentParser(description='Learned BC policy for the Pick-to-Delivery route')
    ap.add_argument('mode',choices=('train','eval','gazebo-drive','gazebo-deliver','gazebo-full'))
    ap.add_argument('--model',type=Path,default=MODEL)
    ap.add_argument('--episodes',type=int,default=80)
    args=ap.parse_args()
    if args.mode=='train':
        p=train(args.model);evaluate(p,args.episodes)
    elif args.mode=='eval':
        evaluate(load(args.model),args.episodes)
    elif args.mode=='gazebo-drive':
        control(load(args.model),PHASES,do_pick=False,do_place=False)
    elif args.mode=='gazebo-deliver':
        control(load(args.model),PHASES,do_pick=True,do_place=True)
    elif args.mode=='gazebo-full':
        # Full task: reset -> EXISTING learned Pick navigator -> recorded Pick
        # -> NEW learned reverse/turn/forward -> recorded Place/Release.
        print('[FULL] reset course, then navigate to Pick with previously trained BC policy', flush=True)
        subprocess.run([sys.executable,str(COURSE/'reset_course.py'),'--apply'],check=True,timeout=110)
        subprocess.run([sys.executable,str(COURSE/'bc_pick_navigation.py'),'gazebo'],check=True,timeout=300)
        control(load(args.model),PHASES,do_pick=True,do_place=True)

if __name__=='__main__':
    main()
