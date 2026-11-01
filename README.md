# Mobile CRANE+ V2 — Behavior Cloning Delivery Demo

Single-box robotic delivery in **ROS 2 Humble + Ignition Gazebo Fortress**.

A trained **Behavior Cloning** navigation policy drives the mobile base to the
pickup location, reverses with the box, turns about 90 degrees, and drives to
the delivery stand. The robot arm uses **prerecorded Pick/Place trajectories**;
Gazebo **DetachableJoint** assists with holding the box during transport.
This demonstration does **not** use reinforcement learning.

## Contents

- `course_v1/bc_full_fixed_delivery.py` — integrated workflow (`check`, `full`, resume modes)
- `course_v1/bc_pick_navigation.py`, `bc_delivery_navigation.py` — navigation policies
- `course_v1/*.npz` — trained BC weights
- `replay_trajectory.py`, `place_arm_full_test.py` — prerecorded arm motions
- `recordings/*.csv` — required trajectory data
- `course_v1/course_world.sdf` and `models/*.sdf` — simulation scene/assets
- `mobile_crane.absolute_mesh.sdf` — integrated robot SDF with low-friction caster

## Running

Requires an existing ROS 2 Humble / Gazebo Fortress installation with CRANE+ V2
robot description, meshes, `ign_ros2_control`, controllers, and the ROS-Gazebo bridge.
Some SDF files reference absolute paths from the author's Docker environment;
**this repository is not yet a turnkey clean-machine installation**.

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
python3 /workspace/mobile_crane/course_v1/bc_full_fixed_delivery.py check
python3 -u /workspace/mobile_crane/course_v1/bc_full_fixed_delivery.py full
```

If an experiment stops while the box is attached, do not immediately soft-reset;
inspect the joint state before restarting.

## Demo

YouTube: _Add the published video URL here._

## Acknowledgements

CRANE+ V2 and related assets may be supplied by external ROS packages; please
obtain external dependencies from their original sources and follow their licenses.
