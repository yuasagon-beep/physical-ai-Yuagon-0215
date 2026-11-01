#!/usr/bin/env python3
"""Generate a visually guided single-box Gazebo Fortress course.

Does not change a running world. Run inside the user's ROS2/Gazebo container.
"""
import copy
import csv
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

BASE = Path(os.environ.get('MOBILE_CRANE_DIR', '/workspace/mobile_crane'))
OUT = BASE / 'course_v1'
START = (-0.55, 0.0, 0.0)
PICK = (0.0, 0.0, 0.0)
TURN = START
DROP = (-0.55, 1.0, math.pi / 2)
PICK_STAND = (0.275, 0.0)
PICK_BOX = (0.240, 0.0, 0.171)
PICK_TOP = 0.145
DROP_LOCAL = (0.137, 0.250)
DROP_SIZE = 0.08  # Keep dimensions from successful physical experiment
DROP_TOP = 0.1815  # Baseline; must be rechecked against the actual release pose


def pos_relative(pose, dx, dy):
    x, y, theta = pose
    return (x + math.cos(theta)*dx - math.sin(theta)*dy,
            y + math.sin(theta)*dx + math.cos(theta)*dy)

DROP_XY = pos_relative(DROP, *DROP_LOCAL)


def f(x):
    return f'{x:.7f}'


def pose_str(x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
    return ' '.join(f(v) for v in (x,y,z,roll,pitch,yaw))


def child(parent, tag, val=None, **attrs):
    e = ET.SubElement(parent, tag, attrs)
    if val is not None:
        e.text = str(val)
    return e


def colored_material(parent, rgb, transparency=0.0):
    child(parent, 'transparency', f(transparency))
    child(parent, 'cast_shadows', 'false')
    mat = child(parent, 'material')
    rgba = ' '.join(f(v) for v in (*rgb, 1.0))
    child(mat, 'ambient', rgba)
    child(mat, 'diffuse', rgba)


def primitive_visual(link, name, xyz, dims, rgb, transparency=0.0, kind='box', yaw=0.0):
    v = child(link, 'visual', name=name)
    child(v, 'pose', pose_str(*xyz, yaw=yaw))
    g = child(v, 'geometry')
    if kind == 'box':
        child(child(g, 'box'), 'size', ' '.join(f(v) for v in dims))
    elif kind == 'cylinder':
        c = child(g, 'cylinder')
        child(c, 'radius', f(dims[0]))
        child(c, 'length', f(dims[1]))
    else:
        raise ValueError(kind)
    colored_material(v, rgb, transparency)
    return v


def make_scenery():
    model = ET.Element('model', name='course_graphics')
    child(model, 'static', 'true')
    link = child(model, 'link', name='paint')
    # Color pads (visuals only; no collisions)
    pads = [(START,'start',(0.18,0.49,1.0)),
            (PICK,'pick',(1.0,0.67,0.14)),
            (DROP,'drop',(0.68,0.34,0.95))]
    for pose, label, color in pads:
        primitive_visual(link, f'{label}_pad', (pose[0],pose[1],0.004),
                         (0.24,0.003), color, 0.48,'cylinder')
        # Target heading: small arrow printed on ground
        fx, fy = math.cos(pose[2]), math.sin(pose[2])
        primitive_visual(link, f'{label}_arrow',
                         (pose[0]+fx*.12, pose[1]+fy*.12, .008),
                         (.14,.025,.002),color, .05,'box',pose[2])
    # Dashed lane boundaries: east-west to pick, north-south to drop
    idx=0
    for y in (-0.36,0.36):
        for x in np.arange(-0.72, 0.11, .145):
            primitive_visual(link, f'hborder_{idx}', (float(x),y,.007),
                             (.075,.016,.002),(.91,.94,.96))
            idx+=1
    for x in (-0.91,-0.19):
        for y in np.arange(0.09,1.18,.145):
            primitive_visual(link, f'vborder_{idx}',(x,float(y),.007),
                             (.016,.075,.002),(.91,.94,.96))
            idx+=1
    # Direction markers along both lane segments
    for i,x in enumerate((-0.38,-0.23,-0.08)):
        primitive_visual(link, f'heading_pick_{i}',(x,-.22,.007),
                         (.08,.012,.002),(.99,.75,.22))
    for i,y in enumerate((.2,.5,.8)):
        primitive_visual(link, f'heading_drop_{i}',(-.72,y,.007),
                         (.012,.08,.002),(.7,.45,1.0))
    # A visibly readable dotted "turn here" marker near the start
    for i in range(12):
        a = math.tau * i/12
        primitive_visual(link,f'turn_ring_{i}',
                         (START[0]+.32*math.cos(a),START[1]+.32*math.sin(a),.007),
                         (.035,.035,.002),(.3,.85,.95),.12)
    return model


def make_stand():
    model = ET.Element('model',name='delivery_stand')
    child(model,'static','true')
    child(model,'pose', pose_str(*DROP_XY,0,0,0,DROP[2]))
    link = child(model,'link',name='body')
    child(link,'pose',pose_str(0,0,DROP_TOP/2))
    for kind in ('collision','visual'):
        node = child(link,kind,name=kind)
        child(child(node,'geometry'),'box')
        child(node.find('geometry').find('box'),'size',
              f'{f(DROP_SIZE)} {f(DROP_SIZE)} {f(DROP_TOP)}')
        if kind == 'visual':
            colored_material(node,(0.52,0.34,0.94))
    return model


def origin(element):
    if element is None:
        return np.eye(4)
    xyz=[float(v) for v in element.get('xyz','0 0 0').split()]
    rpy=[float(v) for v in element.get('rpy','0 0 0').split()]
    x,y,z=rpy
    cx,cy,cz=map(math.cos,rpy)
    sx,sy,sz=map(math.sin,rpy)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz=np.array([[cz,-sz,0],[sz,cz,0], [0,0,1]])
    T=np.eye(4)
    T[:3,:3]=Rz@Ry@Rx
    T[:3,3]=xyz
    return T


def rot_axis(axis,theta):
    axis=np.asarray(axis,dtype=float)
    axis=axis/np.linalg.norm(axis)
    c,s=math.cos(theta),math.sin(theta)
    K=np.array([[0,-axis[2],axis[1]],
                [axis[2],0,-axis[0]],
                [-axis[1],axis[0],0]])
    T=np.eye(4)
    T[:3,:3]=np.eye(3)*c+(1-c)*np.outer(axis,axis)+s*K
    return T


def matrix_rpy(T):
    R=T[:3,:3]
    pitch=math.atan2(-R[2,0], math.hypot(R[0,0],R[1,0]))
    if abs(math.cos(pitch))>1e-7:
        roll=math.atan2(R[2,1],R[2,2])
        yaw=math.atan2(R[1,0],R[0,0])
    else:
        roll=0
        yaw=math.atan2(-R[0,1],R[1,1])
    return tuple(T[:3,3])+ (roll,pitch,yaw)


def all_link_poses(urdf,angles):
    links = {e.get('name'):e for e in urdf.findall('link')}
    poses={'chassis_link':np.eye(4)}
    joints=urdf.findall('joint')
    unresolved=joints[:]
    for _ in range(len(joints)+1):
        if not unresolved: break
        remain=[]
        for joint in unresolved:
            parent=joint.find('parent').get('link')
            dest=joint.find('child').get('link')
            if parent not in poses:
                remain.append(joint)
                continue
            T=poses[parent] @ origin(joint.find('origin'))
            if joint.get('type') in ('revolute','continuous'):
                a=joint.find('axis')
                axis=[float(v) for v in a.get('xyz','1 0 0').split()] if a is not None else [1,0,0]
                T=T @ rot_axis(axis,angles.get(joint.get('name'),0.0))
            poses[dest]=T
        unresolved=remain
    if unresolved or set(poses)!=set(links):
        raise RuntimeError(f'URDF link graph incomplete: missing {set(links)-set(poses)}')
    return links,poses


def geometry_to_sdf(visual, mesh_root):
    geometry=visual.find('geometry')
    if geometry is None: return None
    out=ET.Element('geometry')
    if geometry.find('mesh') is not None:
        src=geometry.find('mesh')
        fn=src.get('filename')
        if not fn: return None
        prefix='package://crane_plus_description/'
        if fn.startswith(prefix):
            file_path=mesh_root / fn[len(prefix):]
            if not file_path.is_file():
                raise RuntimeError(f'Ghost mesh not found: {file_path}')
            fn=file_path.as_uri()
        elif fn.startswith('file://'):
            pass
        else:
            raise RuntimeError('Unsupported mesh URI: '+fn)
        mesh=child(out,'mesh')
        child(mesh,'uri',fn)
        if src.get('scale'): child(mesh,'scale',src.get('scale'))
    elif geometry.find('box') is not None:
        src=geometry.find('box')
        child(child(out,'box'),'size',src.get('size'))
    elif geometry.find('cylinder') is not None:
        src=geometry.find('cylinder')
        obj=child(out,'cylinder')
        child(obj,'radius',src.get('radius'))
        child(obj,'length',src.get('length'))
    elif geometry.find('sphere') is not None:
        src=geometry.find('sphere')
        child(child(out,'sphere'),'radius',src.get('radius'))
    else:
        return None
    return out


def make_ghost(urdf, mesh_root, name, worldpose, angles, color):
    links, poses=all_link_poses(urdf,angles)
    model=ET.Element('model', name=name)
    child(model,'static','true')
    child(model,'pose',pose_str(*worldpose[:2],0,0,0,worldpose[2]))
    link=child(model,'link',name='all_visuals')
    count=0
    for link_name,urdf_link in links.items():
        T=poses[link_name]
        for vi,urdf_vis in enumerate(urdf_link.findall('visual')):
            geometry=geometry_to_sdf(urdf_vis,mesh_root)
            if geometry is None: continue
            visual=child(link,'visual',name=f'{link_name}_{vi}')
            vpose=T @ origin(urdf_vis.find('origin'))
            child(visual,'pose',pose_str(*matrix_rpy(vpose)))
            visual.append(geometry)
            colored_material(visual,color,.73)
            count+=1
    if count<7:
        raise RuntimeError(f'Ghost has too few visuals: {count}')
    return model,count


def existing_model(path,name,pose):
    model=copy.deepcopy(ET.parse(path).getroot().find('model'))
    if model is None: raise RuntimeError(f'No SDF model: {path}')
    model.set('name',name)
    prior=model.find('pose')
    if prior is not None: model.remove(prior)
    model.insert(0,ET.Element('pose'))
    model.find('pose').text=pose_str(*pose)
    return model


def validate_existing_geometry(pickup_file, cube_file):
    """Fail early if known bounding-box assumptions do not match the real SDF."""
    def dims(file):
        root=ET.parse(file).getroot()
        n=root.find('.//collision/geometry/box/size')
        if n is None:
            print('WARNING: Could not read box collision size:',file)
            return None
        return tuple(float(v) for v in n.text.split())
    stand=dims(pickup_file)
    cube=dims(cube_file)
    if stand is not None:
        if len(stand)!=3: raise RuntimeError('Invalid pickup stand box geometry')
        gap = PICK_STAND[0] - stand[0]/2 - .18
        print('Pickup stand size:',stand,'; chassis front gap:',round(gap,4),'m')
        if gap < .012:
            raise RuntimeError('Pickup stand is too close to chassis; increase stop distance / adapt stand')
        if cube is not None and abs(PICK_STAND[0]-PICK_BOX[0])+cube[0]/2 > stand[0]/2 + .005:
            raise RuntimeError('Cube would not fit on the pickup stand in X')
    if cube is not None and any(abs(x-.05)>.005 for x in cube):
        raise RuntimeError(f'Unexpected product box dimensions: {cube}')


def generate():
    OUT.mkdir(parents=True,exist_ok=True)
    urdf_file=BASE/'mobile_crane.urdf'
    world_file=BASE/'mobile_world.sdf'
    pickup_file=BASE/'models/pickup_stand.sdf'
    cube_file=BASE/'models/product_cube.sdf'
    traj_file=BASE/'recordings/place_trajectory.csv'
    for file in (urdf_file,world_file,pickup_file,cube_file,traj_file):
        if not file.is_file(): raise FileNotFoundError(file)
    validate_existing_geometry(pickup_file, cube_file)
    mesh_root=Path(os.environ.get('CRANE_DESCRIPTION_DIR', '/home/ubuntu/airobot_ws/src/crane_plus/crane_plus_description'))
    if not mesh_root.is_dir():
        try:
            from ament_index_python.packages import get_package_share_directory
            mesh_root=Path(get_package_share_directory('crane_plus_description'))
        except Exception:
            raise RuntimeError('crane_plus_description meshes not found')
    base=ET.parse(world_file)
    world=base.getroot().find('world')
    if world is None: raise RuntimeError('Expected <sdf><world> in mobile_world.sdf')
    if world.get('name')!='mobile_world':
        raise RuntimeError(f'Expected world mobile_world, found {world.get("name")}')
    present={m.get('name') for m in world.findall('model')}
    for n in ('pickup_stand','product_cube','delivery_stand','course_graphics','ghost_pick','ghost_drop','mobile_crane'):
        if n in present: raise RuntimeError(f'Base world already contains model {n}; refusing duplicate')
    urdf=ET.parse(urdf_file).getroot()
    home={f'crane_plus_joint{i}':v for i,v in enumerate([0,-1.16,-2.01,-.73],1)}
    home['crane_plus_joint_hand']=0.0
    with traj_file.open(newline='') as fh:
        data=list(csv.DictReader(fh))
    release_rows=[r for r in data if float(r['relative_time_sec'])<4.184]
    if not release_rows: raise RuntimeError('No pre-release trajectory data')
    final=release_rows[-1]
    release={f'crane_plus_joint{i}':float(final[f'crane_plus_joint{i}']) for i in range(1,5)}
    release['crane_plus_joint_hand']=.105
    ghost_pick,n1=make_ghost(urdf,mesh_root,'ghost_pick',PICK,home,(.12,.86,1.0))
    ghost_drop,n2=make_ghost(urdf,mesh_root,'ghost_drop',DROP,release,(.78,.40,1.0))
    models=[
        make_scenery(),
        existing_model(pickup_file,'pickup_stand',(*PICK_STAND,0,0,0,0)),
        existing_model(cube_file,'product_cube',(*PICK_BOX,0,0,0)),
        make_stand(),ghost_pick,ghost_drop
    ]
    for m in models: world.append(m)
    target=OUT/'course_world.sdf'
    ET.indent(base,space='  ')
    base.write(target,encoding='utf-8',xml_declaration=True)
    state={'world':'mobile_world',
           'start': {'x':START[0],'y':START[1],'yaw':START[2]},
           'pick_stop':{'x':PICK[0],'y':PICK[1],'yaw':PICK[2]},
           'turn':{'x':TURN[0],'y':TURN[1],'yaw':TURN[2]},
           'drop_stop':{'x':DROP[0],'y':DROP[1],'yaw':DROP[2]},
           'pickup_stand':{'x':PICK_STAND[0],'y':PICK_STAND[1],'top_z':PICK_TOP},
           'product_cube':{'x':PICK_BOX[0],'y':PICK_BOX[1],'z':PICK_BOX[2]},
           'delivery_stand':{'x':DROP_XY[0],'y':DROP_XY[1],'top_z':DROP_TOP,'width':DROP_SIZE},
           'route': ['W (start→pick)','P','S (pick→start)','A ~90°','W (start→drop)','R'],
           'ghost_visual_counts':{'pick':n1,'drop':n2},
           'notes':'Keys cannot be counted exactly until drive speed/pulse is calibrated. Ghosts are static non-colliding visuals. Arm not included in world; spawn real robot separately.'}
    (OUT/'course_config.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    print('Generated',target)
    print('Ghost geometry counts:',n1,n2)
    print('Pickup stand',PICK_STAND,'cube',PICK_BOX,'drop stand',tuple(round(v,4) for v in DROP_XY))
    print('Robot spawn position',START)
    print('No running Gazebo world was modified')
    print('Next: ign sdf -k',target)


if __name__=='__main__':
    generate()
