#!/usr/bin/env python3
"""Read-only key-pulse measurement. Operator uses existing keyboard controller in another terminal."""
import math
import re
import subprocess


def pose():
    p=subprocess.run(['ign','model','-m','mobile_crane','--pose'],
                     capture_output=True,text=True,timeout=15,check=True)
    a=re.findall(r'\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]',p.stdout)
    if len(a)<2: raise RuntimeError(p.stdout)
    xyz=list(map(float,a[0])); rot=list(map(float,a[1]))
    return xyz[0],xyz[1],rot[2]


def diff_angle(a,b):
    return math.atan2(math.sin(b-a), math.cos(b-a))

print('This tool only reads poses. It does NOT drive the robot.')
print('Run it in a separate terminal while drive_mobile_crane.py is open.')
input('Stop at the START marker. Press Enter here when ready...')
x0,y0,t0=pose()
print('Starting pose:',(x0,y0,t0))
input('Press W exactly once in the driving terminal, wait until stopped, then press Enter here...')
x1,y1,t1=pose()
d=math.cos(t0)*(x1-x0)+math.sin(t0)*(y1-y0)
print('Measured forward movement per W:',round(d,5),'m')
input('Press A exactly once in the driving terminal, wait until stopped, then press Enter here...')
x2,y2,t2=pose()
a=diff_angle(t1,t2)
print('Measured left turn per A:',round(a,5),'rad =',round(math.degrees(a),3),'degrees')
if d > .005 and a > .01:
    print('Approximate NOMINAL route, before final correction:')
    print('W x',round(.55/d),'→ P → S x',round(.55/d),'→ A x',round((math.pi/2)/a),'→ W x',round(1.0/d),'→ R')
else:
    print('Measured pulses were unexpectedly small; check terminal focus and Gazebo state.')
print('Counts are only a first estimate: compare the physical robot with the full-size ghost at both stops.')
