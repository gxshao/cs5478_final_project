import argparse
from genericpath import getctime
from math import nan
from os import curdir, strerror
from hybrid_planner import *
from motion_planner import *
import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import *
import pyglet
import time
# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map2_1", type=str)
parser.add_argument('--seed', '-s', default=2, type=int)
parser.add_argument('--start-tile', '-st', default="3,6", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="7,1", type=str, help="two numbers separated by a comma")
args = parser.parse_args()

env = DuckietownEnv(
    domain_rand=False,
    max_steps=1500,
    map_name=args.map_name,
    seed=args.seed,
    user_tile_start=args.start_tile,
    goal_tile=args.goal_tile,
    randomize_maps_on_reset=False   
)

obs = env.reset()
env.render()

def motion_predict(pos, angle, action):
    x, y = pos, angle
    for i in range(30):
        vel, angle = action

        # Distance between the wheels
        baseline = env.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = env.k
        k_l = env.k

        # adjusting k by gain and trim
        k_r_inv = (env.gain + env.trim) / k_r
        k_l_inv = (env.gain - env.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / env.radius
        omega_l = (vel - 0.5 * angle * baseline) / env.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, env.limit), -env.limit)
        u_l_limited = max(min(u_l, env.limit), -env.limit)

        vels = np.array([u_l_limited, u_r_limited])

        x, y = _update_pos(x, y, 0.102, vels * env.robot_speed * 1, env.delta_time)
    return x, y

map_img, goal, start_pos = env.get_task_info()
print("start tile:", start_pos, " goal tile:", goal)


# Test
# pos = env.cur_pos
# angle = env.cur_angle
# print("initial pose:", pos, angle)
# print("predcit:", motion_predict(pos, angle, [0.1, 1.57]))
# for i in range(30):
#     env.step([0.1, 1.57])
#     env.render()
# print("ref pose:",env.cur_pos, env.cur_angle)




planner = MotionPlanner(env)
path =planner.astar()
# path = [[7, 7], [6, 7], [5, 7], [5, 6], [5, 5], [4, 5], [3, 5], [3, 4], [3, 3], [2, 3], [1, 3], [1, 2], [1, 1]]

def closest_curve_point(pos, curves, angle=None):
    curve_headings = curves[:, -1, :] - curves[:, 0, :]
    #curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
    
    dir_vec = env.get_dir_vec(angle)
    dot_prods = np.dot(curve_headings, dir_vec)

    cps = curves[np.argmax(dot_prods)]
    t = bezier_closest(cps, pos)
    point = bezier_point(cps, t)
    tangent = bezier_tangent(cps, t)
    return point, tangent

def get_lane_pos2(pos, angle, target_curve):
    point, tangent = closest_curve_point(pos, target_curve, angle)
    if point is None:
        msg = 'Point not in lane: %s' % pos
        raise Exception(msg)
    assert point is not None
    dirVec = env.get_dir_vec(angle)
    dotDir = np.dot(dirVec, tangent)
    dotDir = max(-1, min(1, dotDir))
    posVec = pos - point
    upVec = np.array([0, 1, 0])
    rightVec = np.cross(tangent, upVec)
    signedDist = np.dot(posVec, rightVec)
    angle_rad = math.acos(dotDir)
    if np.dot(dirVec, rightVec) < 0:
        angle_rad *= -1
    angle_deg = np.rad2deg(angle_rad)
    return LanePosition(dist=signedDist, dot_dir=dotDir, angle_deg=angle_deg,
                        angle_rad=angle_rad)
# x1, x2, x3, x0
# def bezier(ja, jb, start, end):
#     p = []
#     for s in range(10):
#         if s %2 == 1:
#             continue
#         t = s / 10
#         p.append([(1-t)*((1-t)*((1-t) * end[0] + t * ja[0]) + t * ((1-t) * ja[0] + t * jb[0])) + t*((1-t) * ((1-t) * ja[0]) + t * ((1-t) * jb[0] + t * start[0])), 0, \
#         (1-t)*((1-t)*((1-t) * end[1] + t * ja[1]) + t*((1-t) * ja[1] + t * jb[1])) + t * ((1-t) * ((1-t) * ja[1] + t * jb[1]) + t * ((1-t) * jb[1] + t * start[1]))])
#     return p

# bezier_curve = bezier([0,1], [0, 1], [0, 0], [1, 1])
# dots = np.array([], np.int32)
# for p in bezier_curve:
#     p[0] += 5.7
#     p[2] += 6.3
#    # dots = np.append(dots, [int((p[0]) * 100), int((p[2]) * 100)])

# print(bezier_curve)

def get_curve(curve, offset, map_img):
    tile = env._get_tile(offset[0], offset[1])
    print("angle match:", current_key, tile['angle'])
    diff = current_key - tile['angle']
    # if diff > 0:
        #clockwise
        # curve = np.rot90(curve, diff)
    # elif diff < 0:
        # anti
        # curve = np.rot90(curve, 4 + diff)
         
    curve *= env.road_tile_size
    mat = gen_rot_matrix(np.array([0, 1, 0]), (tile['angle']) * math.pi / 2)
    curve = np.matmul(curve, mat)
    curve += np.array([(offset[0] + 0.5) * env.road_tile_size, 0, (offset[1] + 0.5) * env.road_tile_size])

    for c in curve:
        dts = np.array([], np.int32)
        for p in c:
            dts = np.append(dts, [int((p[0]) * 100), int((p[2]) * 100)])
        dts = dts.reshape((-1,1,2))
        map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=3)
        cv2.imshow("map", map_img)
        cv2.waitKey(100)
        
        
    return curve

curve_list = {(1,1):np.array([
                [
                    [-0.30, 0, -0.20],
                    [-0.30, 0, 0.20],
                    [-0.20, 0, 0.30],
                    [0.20, 0, 0.30],
                ],
            ]),
            (-1,-1):np.array([
                 [
                    [-0.30, 0, -0.20],
                    [0.20, 0, -0.20],
                    [0.20, 0, -0.20],
                    [0.20, 0, 0.30],
                ],
            ]),
            # E-N
            (1,-1): np.array([
                [
                    [-0.20, 0, -0.20],
                    [0.20, 0, -0.20],
                    [-0.20, 0, -0.20],
                    [-0.20, 0, 0.20],
                ]
            ]),
            # W-S
            (-1,1): np.array([
                [
                     [ 0.2,  0.0  , 0.3],
                     [ 0.2 , 0.0 , -0.2],
                     [ 0.2 , 0.0 , -0.2],
                     [0.3 , 0.0 , -0.2]
                ]
            ])
            }
# print(np.fliplr(curve_list[(1, -1)]))
# print(np.flipud(curve_list[(1, -1)]))
base_curve = np.array([
                [
                    [-0.20, 0, -0.20],
                    [0.20, 0, -0.20],
                    [-0.20, 0, -0.20],
                    [-0.20, 0, 0.20],
                ],])
# np.array(
#     [[[ 0.2 , 0.  , 0.2]
#   [-0.2 , 0. ,  0.2]
#   [-0.2 , 0. ,  0.2]
#   [-0.2 , 0. , -0.2]]]
# )

# for i in range(1, 7):
#     get_curve(base_curve, [i,7 - i], map_img)



# pyglet.app.run()
# p = [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]
# print(path)
# turn_curves = []
# for i in range(1, len(path)):
#     next_point = None
#     if i + 1 < len(path):
#         next_point = path[i + 1]
#     if next_point is None:
#         break
#     if next_point[0] != p[0] and next_point[1] != p[1]:
#         #print('Turning', next_point)
#         local = np.array([p[0] - path[i][0], p[1] - path[i][1]], np.int32)
#         target = np.array([next_point[0] - path[i][0], next_point[1] - path[i][1]], np.int32)
#         key = tuple(local + target)
#         print(key, local ,target)
#         offset = [path[i][0] + 1, path[i][1]]
#         turn_curves.append(get_curve(curve_list[key], offset, map_img))
#         if len(turn_curves) == 2:
#             break
#     p = path[i]

curve_angles = {(1, -1):1, (1, 1):0, (-1,1):3, (-1,-1):2}

total_reward = 0
dts = np.array([], np.int32)


predicted_pos = start_pos
turning_index = 0

turning_curve = None
local_target = None
trig_target = None
current_key = None

pending_step = 0
k_p = 10
k_d = 3
speed = 0.3

def generate_action(robot_theta, target_theta):
    if target_theta == -1:
        return 0
    theta = [{0: 0, 90: 1, 180: 1, 270: -1},
             {0: -1, 90: 0, 180: 1, 270: 1},
             {0: 1, 90: -1, 180: 0, 270: 1},
             {0: 1, 90: 1, 180: -1, 270: 0}]

    pose_set = {0: 0, 90: 1, 180: 2, 270: 3}
    robot_theta = robot_theta * 90 % 360
    target_theta = target_theta * 90 % 360
    return theta[pose_set[robot_theta]][target_theta]

target_theta = {(1,0):0,(0,1):3,(-1,0):2,(0,-1):1}

hand_speed = 0
hand_steering = 0
while True:
    time.sleep(0.01)
    lane_pose = None
    distance_to_road_center = 0
    angle_from_straight_in_rads = 0

    predicted_pos = [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]
    speed = hand_speed
    steering = hand_steering
    if predicted_pos != trig_target:
        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        k_p = 80
        k_d = 10
        speed = 0.8
        
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rads = lane_pose.angle_rad
        steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads
        
    obs, reward, done, info = env.step([speed, steering])
    total_reward += reward
    
    predicted_pos = [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]
    # Turning finished
    if predicted_pos != trig_target:
        print("PID is controling")
        local_target = None
        trig_target = None
        turning_curve = None
        
    # I Need To Turn
    try:
        path_index = path.index(predicted_pos)
        if trig_target is None and len(path) > path_index + 2:
            mid = path[path_index + 1]
            next_point = path[path_index + 2]
            if next_point[0] != predicted_pos[0] and next_point[1] != predicted_pos[1]:
                target = np.array([next_point[0] - mid[0], next_point[1] - mid[1]], np.int32)
                # print("I need", curve_angles[key])
                trig_target = mid
                print("I am  controling")
                angle = int(abs(np.rad2deg(env.cur_angle)) % 360)
                suffix = angle % 90
                if suffix > 60:
                    suffix = 1
                else:
                    suffix = 0
                
                angle = math.floor((angle / 90)) + suffix
                t = target_theta[(next_point[0] - trig_target[0], next_point[1] - trig_target[1])]
                c = generate_action(angle, t)
                print(c,angle, t)
                if c > 0:
                    hand_speed = 0.4
                    hand_steering = c * 0.6
                else:
                    hand_speed = 0.2
                    hand_steering = c * 0.8
    except:
        pass
    
    d = [int(env.cur_pos[0] * 100), int(env.cur_pos[2] * 100)]
    dts = np.append(dts,d)
    dts = dts.reshape((-1,1,2))
    map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=3)
    cv2.imshow("map", map_img)
    cv2.waitKey(10)
    env.render()
    if done:
        break

print("done")

pyglet.app.run()