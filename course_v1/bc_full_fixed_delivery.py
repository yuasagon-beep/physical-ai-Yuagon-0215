#!/usr/bin/env python3
"""One-command BC navigation + scripted Pick/Place + Gazebo DetachableJoint.

Tested: static syntax and mock scaffolding only. The user must validate Gazebo operation.
Does not claim arm learning or RL. Run in existing ROS2 Humble + Gazebo Fortress container.
"""
import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

COURSE = Path('/workspace/mobile_crane/course_v1')
BASE = COURSE.parent
sys.path.insert(0, str(COURSE))

TOPIC_STATE = '/course/grip/state'
TOPIC_ATTACH = '/course/grip/attach'
TOPIC_DETACH = '/course/grip/detach'
SERVICE = '/world/mobile_world/entity/system/add'
TOPIC_CMD = '/model/mobile_crane/cmd_vel'
LOG_DIR = COURSE / 'experiment_logs'

# In the SDF, the drive-wheel axle lies ~0.08 m behind the model origin.
AXLE_OFFSET = 0.08

def axle_xy(x, y, yaw):
    return x - AXLE_OFFSET * math.cos(yaw), y - AXLE_OFFSET * math.sin(yaw)


def run(cmd, timeout=80, capture=False):
    print('[RUN]', ' '.join(map(str, cmd)), flush=True)
    cp = subprocess.run([str(x) for x in cmd], text=True,
                        capture_output=capture, timeout=timeout)
    if capture:
        if cp.stdout:
            print(cp.stdout.strip(), flush=True)
        if cp.stderr:
            print(cp.stderr.strip(), file=sys.stderr, flush=True)
    if cp.returncode:
        raise RuntimeError(f'Exit status {cp.returncode}: {cmd}')
    return cp.stdout if capture else ''


def gazebo_pose(name):
    from bc_delivery_navigation import gazebo_pose as impl
    return impl(name)


def msg_topic(topic):
    run(['ign', 'topic', '-t', topic, '-m', 'ignition.msgs.Empty', '-p', ' '],
        timeout=20)
    time.sleep(.6)


def plugin_is_loaded():
    proc = subprocess.run(['ign', 'topic', '-i', '-t', TOPIC_STATE],
                          capture_output=True, text=True, timeout=12)
    return proc.returncode == 0 and 'ignition.msgs.StringMsg' in (proc.stdout + proc.stderr)


def detach_for_reset():
    if not plugin_is_loaded():
        print('[JOINT] No active state publisher; no previous fixed joint detected.')
        return
    print('[JOINT] Previous joint plugin found. Detach before resetting.', flush=True)
    msg_topic(TOPIC_DETACH)
    # This only proves that the request was published, not physical detachment.
    print('[JOINT] Detach requested; reset will separately validate resulting poses.', flush=True)


def attach_after_pick():
    from bc_delivery_navigation import gazebo_pose
    from bc_delivery_navigation import check_running_and_ready
    check_running_and_ready()
    box, _ = gazebo_pose('product_cube')
    base, _ = gazebo_pose('mobile_crane')
    if not .285 < box[2] < .40:
        raise RuntimeError(f'Refusing fixed joint: cube not lifted, z={box[2]:.3f}')
    if math.hypot(base[0], base[1]) > .035:
        raise RuntimeError('Refusing fixed joint: robot not at Pick stop.')
    if plugin_is_loaded():
        print('[JOINT] Existing plugin; request attach.', flush=True)
        msg_topic(TOPIC_ATTACH)
    else:
        identity = subprocess.run(
            ['ign', 'model', '-m', 'mobile_crane', '--pose'],
            capture_output=True, text=True, timeout=16)
        m = re.search(r'Model:\s*\[(\d+)\]', identity.stdout)
        if identity.returncode or not m:
            raise RuntimeError('Robot entity ID unavailable: refusing plugin add')
        robot_id = int(m.group(1))
        xml = (
            '<parent_link>crane_plus_link_hand</parent_link>'
            '<child_model>product_cube</child_model>'
            '<child_link>cube</child_link>'
            f'<attach_topic>{TOPIC_ATTACH}</attach_topic>'
            f'<detach_topic>{TOPIC_DETACH}</detach_topic>'
            f'<output_topic>{TOPIC_STATE}</output_topic>'
        )
        req = (
            f'entity {{ id: {robot_id} }} plugins {{ '
            'name: "ignition::gazebo::systems::DetachableJoint" '
            'filename: "ignition-gazebo-detachable-joint-system" '
            f'innerxml: {json.dumps(xml)} }}'
        )
        out = run(['ign', 'service', '-s', SERVICE,
                   '--reqtype', 'ignition.msgs.EntityPlugin_V',
                   '--reptype', 'ignition.msgs.Boolean',
                   '--timeout', '15000', '--req', req],
                  timeout=26, capture=True)
        if not re.search(r'\bdata:\s*true\b', out):
            raise RuntimeError('Gazebo did not confirm adding DetachableJoint')
        for _ in range(12):
            if plugin_is_loaded():
                break
            time.sleep(.4)
        if not plugin_is_loaded():
            raise RuntimeError('Joint system accepted but status publisher unavailable')
    time.sleep(1.5)
    return anchor_position()


def anchor_position():
    (x,y,z),(_,_,yaw) = gazebo_pose('mobile_crane')
    (bx,by,bz),_ = gazebo_pose('product_cube')
    dx,dy=bx-x,by-y
    return (math.cos(yaw)*dx + math.sin(yaw)*dy,
            -math.sin(yaw)*dx + math.cos(yaw)*dy,
            bz-z)


def check_load(anchor, stage):
    cube, _ = gazebo_pose('product_cube')
    now = anchor_position()
    shift = math.hypot(now[0]-anchor[0], now[1]-anchor[1])
    dz = now[2]-anchor[2]
    if cube[2] < .255 or shift > .035 or abs(dz) > .035:
        raise RuntimeError(f'[{stage}] Box no longer follows robot: '
                           f'z={cube[2]:.3f}, drift={shift:.3f}, dz={dz:.3f}')
    return cube


def ready():
    needed=[COURSE/'bc_pick_navigation.py', COURSE/'bc_pick_policy.npz',
            COURSE/'bc_delivery_navigation.py',COURSE/'bc_delivery_policy.npz',
            COURSE/'reset_course.py',BASE/'replay_trajectory.py',
            BASE/'place_arm_full_test.py']
    absent=[str(p) for p in needed if not p.exists()]
    if absent: raise RuntimeError('Missing required files:\n'+'\n'.join(absent))
    from bc_delivery_navigation import check_running_and_ready
    check_running_and_ready()
    msg=subprocess.run(['ros2','topic','info','/clock'],capture_output=True,text=True,timeout=20)
    if 'Publisher count: 0' in msg.stdout or msg.returncode:
        raise RuntimeError('Clock bridge not connected')
    msg=subprocess.run(['ros2','topic','info',TOPIC_CMD],capture_output=True,text=True,timeout=20)
    m=re.search(r'Subscription count:\s*(\d+)',msg.stdout)
    if not m or int(m.group(1))<1:
        raise RuntimeError('cmd_vel bridge not connected')
    gazebo_pose('mobile_crane')
    gazebo_pose('product_cube')
    print('[READY] Gazebo, controllers, and ROS bridges available',flush=True)


def reset_checked():
    detach_for_reset()
    run([sys.executable,COURSE/'reset_course.py','--apply'],timeout=125)
    (x,y,_),(_,_,theta)=gazebo_pose('mobile_crane')
    cube,_=gazebo_pose('product_cube')
    if abs(x+.55)>.03 or abs(y)>.025 or abs(theta)>.04:
        raise RuntimeError('Reset robot pose incorrect; possible joint remained attached')
    if math.hypot(cube[0]-.24,cube[1])>.022 or not .14<cube[2]<.21:
        raise RuntimeError('Reset cube pose incorrect; abort before AI driving')
    print('[RESET] Verified separated start robot and box poses',flush=True)


def nav_to_pick():
    print('[BC PICK] neural network controls vehicle to Pick',flush=True)
    run([sys.executable,COURSE/'bc_pick_navigation.py','gazebo'],timeout=330)
    pose,rpy=gazebo_pose('mobile_crane')
    if abs(pose[0])>.015 or abs(pose[1])>.020 or abs(rpy[2])>.055:
        raise RuntimeError('Pick navigator reported done, but pose is outside Pick tolerance')


def pick_and_lock():
    print('[PICK] replay recorded arm/gripper action (NOT learned)',flush=True)
    run([sys.executable,BASE/'replay_trajectory.py','pick'],timeout=135)
    cube,_=gazebo_pose('product_cube')
    if cube[2]<.29:
        raise RuntimeError('Pick trajectory did not lift cube')
    anchor=attach_after_pick()
    time.sleep(.8)
    check_load(anchor,'post-attach')
    print('[JOINT] Attachment requested; pose stable. ',
          'Prior 21cm hand-held test confirms this configuration.',flush=True)
    return anchor


def nav_to_delivery(anchor, phases=None):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from bc_delivery_navigation import load, policy, reached, GOALS, PHASES, wrap
    p=load()
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    logfile=LOG_DIR/('bc_fixed_full_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv')
    rclpy.init()
    node=Node('bc_fixed_delivery_controller')
    pub=node.create_publisher(Twist,TOPIC_CMD,10)
    def stop():
        for _ in range(7): pub.publish(Twist());time.sleep(.05)
    def drive(v,w):
        msg=Twist();msg.linear.x=float(v);msg.angular.z=float(w)
        end=time.monotonic()+.40
        while time.monotonic()<end:
            pub.publish(msg)
            time.sleep(.05)
        stop()
    try:
        time.sleep(.5)
        if pub.get_subscription_count()<1:raise RuntimeError('cmd_vel ROS bridge unavailable')
        with logfile.open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['phase','step','x','y','yaw','linear_cmd','angular_cmd','cube_z'])
            for phase in (PHASES if phases is None else phases):
                gx,gy,ga=GOALS[phase]
                print(f'[BC {phase.upper()}] goal=({gx:.3f}, {gy:.3f}, {ga:.3f})',flush=True)
                max_steps={'reverse':85,'turn':60,'forward':130}[phase]
                for i in range(max_steps):
                    (x,y,_),(_,_,yaw)=gazebo_pose('mobile_crane')
                    cube=check_load(anchor, phase)
                    if phase=='reverse' and not (-.69<=x<=.12 and abs(y)<.12 and abs(wrap(yaw))<.23):
                        raise RuntimeError('Reverse lane boundary exceeded')
                    if phase == 'turn':
                        axle_x, axle_y = axle_xy(x, y, yaw)
                        pivot_error = math.hypot(axle_x - (gx - AXLE_OFFSET), axle_y - gy)
                        if pivot_error > .065:
                            raise RuntimeError(f'Turn wheel-axle boundary exceeded: {pivot_error:.3f}m')
                    if phase=='forward' and not (-.68<=x<=-.42 and -.09<=y<=1.08 and abs(wrap(yaw-math.pi/2))<.24):
                        raise RuntimeError('Delivery lane boundary exceeded')
                    # The model origin traces a natural arc during in-place turns;
                    # require the wheel axle, not the model origin, to stay centered.
                    done = (abs(wrap(ga-yaw)) < .016 and pivot_error < .040) if phase == 'turn' else reached(x,y,yaw,phase)
                    if done:
                        stop()
                        print(f'[BC {phase.upper()}] reached: ({x:.4f},{y:.4f},{yaw:.4f})',flush=True)
                        break
                    v,w=policy(x,y,yaw,phase,p)
                    writer.writerow([phase,i,x,y,yaw,v,w,cube[2]])
                    f.flush()
                    if i%5==0:
                        print(f'[BC] {phase} step={i}, x={x:.3f}, y={y:.3f},yaw={math.degrees(yaw):.1f}deg, v={v:.3f},w={w:.3f}',flush=True)
                    drive(v,w)
                else:
                    raise RuntimeError(f'BC {phase} did not reach target within {max_steps} steps')
        check_load(anchor,'end-navigation')
        print('[BC] delivery navigation complete; trajectory saved:',logfile,flush=True)
    finally:
        stop()
        node.destroy_node()
        rclpy.shutdown()


def place_and_release(anchor):
    check_load(anchor,'before-place')
    print('[PLACE] arm-only replay (NOT learned)',flush=True)
    run([sys.executable,BASE/'place_arm_full_test.py'],timeout=155)
    cube,_=gazebo_pose('product_cube')
    stand,_=gazebo_pose('delivery_stand')
    dist=math.hypot(cube[0]-stand[0],cube[1]-stand[1])
    print(f'[PLACE CHECK] cube_z={cube[2]:.3f}, stand_xy={stand[:2]}, error={dist:.3f}m',flush=True)
    if dist>.038 or not .16<cube[2]<.32:
        raise RuntimeError('Box not aligned above stand; leave joint ATTACHED and inspect before releasing')
    print('[RELEASE] detach joint first, then open hand',flush=True)
    msg_topic(TOPIC_DETACH)
    goal=("{trajectory: {joint_names: ['crane_plus_joint_hand'], "
          "points: [{positions: [-0.5235], time_from_start: {sec: 2}}]}}")
    run(['ros2','action','send_goal','/crane_plus_gripper_controller/follow_joint_trajectory',
         'control_msgs/action/FollowJointTrajectory',goal],timeout=65)
    time.sleep(3)
    cube,_=gazebo_pose('product_cube')
    dist=math.hypot(cube[0]-stand[0],cube[1]-stand[1])
    print(f'[FINAL] cube={cube}, xy_error={dist:.3f}',flush=True)
    if dist>.068 or not .15<cube[2]<.24:
        raise RuntimeError('Box release result is outside stand acceptance zone')
    print('******** BC PICK + FIXED JOINT + BC DELIVERY + PLACE SUCCEEDED ********',flush=True)


def resume_current():
    # Continue from the user's experimentally verified attached pose after 5 S pulses.
    if not plugin_is_loaded():
        raise RuntimeError('DetachableJoint system is not present. Cannot resume.')
    xyz,rpy=gazebo_pose('mobile_crane')
    cube,_=gazebo_pose('product_cube')
    if not (-.68<xyz[0]<.045 and abs(xyz[1])<.045 and abs(rpy[2])<.055):
        raise RuntimeError('Robot is not in the reverse corridor facing +X; use check first')
    if cube[2]<.285:
        raise RuntimeError('Box is not lifted; refusing resume')
    anchor=anchor_position()
    print('[RESUME] Using already fixed box at current pose:',xyz,flush=True)
    print('[RESUME] BC drives remaining reverse, 90deg turn, forward',flush=True)
    nav_to_delivery(anchor)
    place_and_release(anchor)


def resume_from_midturn():
    if not plugin_is_loaded():
        raise RuntimeError('DetachableJoint state topic not present: refusing resume')
    (x,y,_),(_,_,yaw) = gazebo_pose('mobile_crane')
    if not (-.70 < x < -.42 and -.07 < y < .16 and .10 < yaw < 1.70):
        raise RuntimeError(f'Not in safe left-turn corridor: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}')
    anchor = anchor_position()
    time.sleep(1.0)
    check_load(anchor, 'pre-resume')
    print('[RESUME TURN] Continuing current held-box turn; no reset or new Pick.', flush=True)
    nav_to_delivery(anchor, phases=('turn', 'forward'))
    place_and_release(anchor)



def resume_from_delivery():
    # Continue a completed delivery navigation without resetting, driving, or Pick.
    if not plugin_is_loaded():
        raise RuntimeError('DetachableJoint state topic missing; refusing Place resume')
    (x, y, _), (_, _, yaw) = gazebo_pose('mobile_crane')
    cube, _ = gazebo_pose('product_cube')
    stand, _ = gazebo_pose('delivery_stand')
    err_yaw = math.atan2(math.sin(yaw - math.pi / 2),
                         math.cos(yaw - math.pi / 2))
    if math.hypot(x + .55, y - 1.0) > .040 or abs(err_yaw) > .085:
        raise RuntimeError(f'Robot is not at delivery stop: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}')
    if not (.285 < cube[2] < .39):
        raise RuntimeError(f'Box is not at held height: z={cube[2]:.3f}')
    distance = math.hypot(cube[0] - stand[0], cube[1] - stand[1])
    if distance > .24:
        raise RuntimeError(f'Box too far from delivery stand: {distance:.3f} m')
    anchor = anchor_position()
    time.sleep(1.0)
    check_load(anchor, 'pre-place-resume')
    print('[RESUME PLACE] Verified stopped delivery pose and stable held box.', flush=True)
    print('[RESUME PLACE] Place -> alignment check -> detach -> open -> verify.', flush=True)
    place_and_release(anchor)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=('check','resume','resume-turn','resume-place','full'))
    args=ap.parse_args()
    if args.mode=='check':
        ready()
        print('robot',gazebo_pose('mobile_crane'))
        print('cube ',gazebo_pose('product_cube'))
        print('joint plugin loaded:',plugin_is_loaded())
        return
    ready()
    if args.mode=='resume':
        resume_current()
        return
    if args.mode=='resume-turn':
        resume_from_midturn()
        return
    if args.mode=='resume-place':
        resume_from_delivery()
        return
    print('[FULL] BC driving is learned; arm Pick/Place is recorded trajectory;',
          ' fixed joint is an explicit simulation assist. NO RL.',flush=True)
    reset_checked()
    nav_to_pick()
    anchor=pick_and_lock()
    nav_to_delivery(anchor)
    place_and_release(anchor)


if __name__=='__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        print(f'\n[ABORT] {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        print('[ABORT] Simulation left intact for inspection. If box is attached, do NOT soft reset without detach.',file=sys.stderr)
        sys.exit(1)
