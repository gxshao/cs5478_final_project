import argparse
from ctypes import alignment
from genericpath import getctime
from logging import FATAL
from math import nan
from os import curdir, spawnlpe, strerror
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
parser.add_argument('--max_steps', type=int, default=5000, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map1_1", type=str)
parser.add_argument('--seed', '-s', default=0, type=int)
parser.add_argument('--start-tile', '-st', default="0,1", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="70,1", type=str, help="two numbers separated by a comma")
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

env.render()


map_img, goal, start_pos = env.get_task_info()
print("start tile:", start_pos, " goal tile:", goal)
map_img = cv2.resize(map_img, None, fx=0.5,fy=0.5)
curve_angles = {(1, -1):1, (1, 1):0, (-1,1):3, (-1,-1):2}
planner = MotionPlanner(env)
path =planner.astar()
print(path)
total_reward = 0
dts = np.array([], np.int32)


predicted_pos = start_pos
trig_target = None

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

def get_vw(m, t):
    angle = int(abs(np.rad2deg(env.cur_angle)) % 360)
    if env.cur_angle < 0:
        angle *= 3
    suffix = angle % 90
    if suffix > 60:
        suffix = 1
    else:
        suffix = 0
    
    angle = math.floor((angle / 90)) + suffix
    t = target_theta[(t[0] - m[0], t[1] - m[1])]
    c = generate_action(angle, t)
    v, w = 0, 0
    if c > 0:
        v = 1
        w = c * 5
    else:
        v = 1
        w = c * 8
    return v, w

target_theta = {(1,0):0,(0,1):3,(-1,0):2,(0,-1):1}


actions = []
## Initial pose calibration
lane_alignment = False
direction_alignment = False
opposite_aligment = False

reward = 0
predicted_pos = [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]
t = target_theta[(path[1][0] - predicted_pos[0], path[1][1] - predicted_pos[1])]
tile = env._get_tile(predicted_pos[0], predicted_pos[1])

rads_threshold = 0.2
kd = tile['kind']

if kd == 'curve_right':
    rads_threshold = 0.9
speed  = 0
steering = 0

hand_speed = 0
hand_steering = 0

initial_reward = 0
while True:
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad
    
    tmp_pos = [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]
    if tmp_pos == path[1]:
        trig_target = path[1]
        hand_speed, hand_steering = get_vw(trig_target, path[2])
        break
    # print(steering, angle_from_straight_in_rads)
    if not lane_alignment and angle_from_straight_in_rads < rads_threshold \
        and angle_from_straight_in_rads > -rads_threshold:
        print("Done lane_alignment")
        lane_alignment = True
    
    angle = int(abs(np.rad2deg(env.cur_angle)) % 360)

    suffix = angle % 90
    if suffix > 60:
        suffix = 1
    else:
        suffix = 0
                
    angle = math.floor((angle / 90)) + suffix
    c = generate_action(angle, t)
    # print(c, angle, t)

    if not lane_alignment:
        speed = 0
        steering = angle_from_straight_in_rads * 10
        print("Lane alignment")
    elif lane_alignment and not direction_alignment:
        if c != 0:
            speed = 0
            steering = 10
            print("Tuning distance_to_road_center")
        elif c == 0:
            direction_alignment = True
            print("Done Direction")
    elif lane_alignment and direction_alignment and not opposite_aligment:
        print("distance :", distance_to_road_center)
        if distance_to_road_center < -0.3:
            speed = 1
            steering = -6
            print("Turn for correct lane")
        elif c != 0:
            pass
            direction_alignment = False
        else:
            print("Done Calibration, distance:", distance_to_road_center)
            opposite_aligment = True
            break
    else:
        break
    actions.append([speed, steering])
    obs, reward, done, info = env.step([speed, steering])
    initial_reward += reward
    env.render()
    print(reward)
    predicted_pos = [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]
    if done:
        break

# pyglet.app.run()

while True:
    lane_pose = None
    distance_to_road_center = 0
    angle_from_straight_in_rads = 0

    predicted_pos = [round(env.cur_pos[0], 1), round(env.cur_pos[2], 1)]

    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    k_p = 240
    k_d = 85
    speed = 1
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad
    steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads
    # print("PID controling", steering)

    
    ###########################ACTION#######################3

    
    predicted_pos = [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]
    # Turning finished
    if predicted_pos != trig_target and trig_target is not None:
        print("PID is controling")  
        local_target = None
        trig_target = None
    elif predicted_pos == trig_target:
        tile = env._get_tile(predicted_pos[0], predicted_pos[1])
        if tile['kind'] != 'curve_right' and tile['kind'] != 'curve_left':
            print("I am  controling")
            speed = hand_speed
            steering = hand_steering
        else:
            print("Let it control")
        
    try:
        path_index = path.index(predicted_pos)
        if len(path) > path_index + 2:
            mid = path[path_index + 1]
            next_point = path[path_index + 2]
            if next_point[0] != predicted_pos[0] \
                and next_point[1] != predicted_pos[1]:
                trig_target = mid
                hand_speed, hand_steering = get_vw(trig_target, next_point)
    except:
        pass
    # print(predicted_pos, trig_target)
    actions.append([speed, steering])
    obs, reward, done, info = env.step([speed, steering])
    total_reward += reward
    d = [int(env.cur_pos[0] * 50), int(env.cur_pos[2] * 50)]
    dts = np.append(dts,d)
    dts = dts.reshape((-1,1,2))
    map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=3)
    cv2.imshow("map", map_img)
    cv2.waitKey(10)
    env.render()
    # print(reward)
    if done or (goal == [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]):
        break

predicted_pos = [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]
np.savetxt(f'/home/marshall/Desktop/duckietown/{args.map_name}_seed{args.seed}_start_{args.start_tile[0]},{args.start_tile[1]}_goal_{args.goal_tile[0]},{args.goal_tile[1]}.txt',
           actions, delimiter=',')

print("done", predicted_pos, total_reward, "initial", initial_reward)
pyglet.app.run()