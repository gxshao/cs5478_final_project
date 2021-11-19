import argparse
from ctypes import alignment
from genericpath import getctime
from logging import FATAL, NullHandler
from math import nan
from os import curdir, strerror
# from hybrid_planner import *
# from motion_planner import *
import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import *
import pyglet
import time
import torch
from PIL import Image
from nnModel import predict

# declare the arguments
parser = argparse.ArgumentParser()

# python generate_motions.py --max_steps 5000 --map-name map3_0 --seed 0 --start-tile 5,7 --goal-tile 2,2


# Do not change this
parser.add_argument('--max_steps', type=int, default=5000, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map1_1", type=str)
parser.add_argument('--seed', '-s', default=0, type=int)
parser.add_argument('--start-tile', '-st', default="0,1", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="70,1", type=str, help="two numbers separated by a comma")
args = parser.parse_args()

goals = {"map1_0":{"seed":[1],"start":[0,1],"goal":[5,1],"path":[[0,1],[1,1],[2,1],[3,1],[4,1],[5,1]]},"map1_1":{"seed":[0],"start":[0,1],"goal":[70,1],"path":[[0,1],[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1],[10,1],[11,1],[12,1],[13,1],[14,1],[15,1],[16,1],[17,1],[18,1],[19,1],[20,1],[21,1],[22,1],[23,1],[24,1],[25,1],[26,1],[27,1],[28,1],[29,1],[30,1],[31,1],[32,1],[33,1],[34,1],[35,1],[36,1],[37,1],[38,1],[39,1],[40,1],[41,1],[42,1],[43,1],[44,1],[45,1],[46,1],[47,1],[48,1],[49,1],[50,1],[51,1],[52,1],[53,1],[54,1],[55,1],[56,1],[57,1],[58,1],[59,1],[60,1],[61,1],[62,1],[63,1],[64,1],[65,1],[66,1],[67,1],[68,1],[69,1],[70,1]]},"map1_2":{"seed":[2],"start":[2,1],"goal":[21,1],"path":[[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1],[10,1],[11,1],[12,1],[13,1],[14,1],[15,1],[16,1],[17,1],[18,1],[19,1],[20,1],[21,1]]},"map1_3":{"seed":[6],"start":[5,1],"goal":[65,1],"path":[[5,1],[6,1],[7,1],[8,1],[9,1],[10,1],[11,1],[12,1],[13,1],[14,1],[15,1],[16,1],[17,1],[18,1],[19,1],[20,1],[21,1],[22,1],[23,1],[24,1],[25,1],[26,1],[27,1],[28,1],[29,1],[30,1],[31,1],[32,1],[33,1],[34,1],[35,1],[36,1],[37,1],[38,1],[39,1],[40,1],[41,1],[42,1],[43,1],[44,1],[45,1],[46,1],[47,1],[48,1],[49,1],[50,1],[51,1],[52,1],[53,1],[54,1],[55,1],[56,1],[57,1],[58,1],[59,1],[60,1],[61,1],[62,1],[63,1],[64,1],[65,1]]},"map1_4":{"seed":[5],"start":[50,1],"goal":[90,1],"path":[[50,1],[51,1],[52,1],[53,1],[54,1],[55,1],[56,1],[57,1],[58,1],[59,1],[60,1],[61,1],[62,1],[63,1],[64,1],[65,1],[66,1],[67,1],[68,1],[69,1],[70,1],[71,1],[72,1],[73,1],[74,1],[75,1],[76,1],[77,1],[78,1],[79,1],[80,1],[81,1],[82,1],[83,1],[84,1],[85,1],[86,1],[87,1],[88,1],[89,1],[90,1]]},"map2_0":{"seed":[1],"start":[7,7],"goal":[1,1],"path":[[7,7],[6,7],[5,7],[5,6],[5,5],[4,5],[3,5],[3,4],[3,3],[2,3],[1,3],[1,2],[1,1]]},"map2_1":{"seed":[2],"start":[3,6],"goal":[7,1],"path":[[3,6],[3,5],[3,4],[3,3],[4,3],[5,3],[5,2],[5,1],[6,1],[7,1]]},"map2_2":{"seed":[5],"start":[1,6],"goal":[3,4],"path":[[1,6],[1,5],[2,5],[3,5],[3,4]]},"map2_3":{"seed":[4],"start":[1,2],"goal":[5,4],"path":[[1,2],[1,3],[2,3],[3,3],[4,3],[5,3],[5,4]]},"map2_4":{"seed":[4],"start":[7,4],"goal":[4,7],"path":[[7,4],[7,5],[6,5],[5,5],[5,6],[5,7],[4,7]]},"map3_0":{"seed":[1],"start":[5,7],"goal":[2,2],"path":[[5,7],[5,6],[5,5],[5,4],[4,4],[3,4],[2,4],[2,3],[2,2]]},"map3_1":{"seed":[2],"start":[5,11],"goal":[1,7],"path":[[5,11],[5,10],[5,9],[5,8],[4,8],[3,8],[2,8],[1,8],[1,7]]},"map3_2":{"seed":[3],"start":[10,5],"goal":[7,11],"path":[[10,5],[11,5],[11,6],[11,7],[10,7],[10,8],[10,9],[9,9],[8,9],[8,10],[8,11],[7,11]]},"map3_3":{"seed":[4],"start":[2,4],"goal":[9,1],"path":[[2,4],[3,4],[4,4],[5,4],[6,4],[7,4],[7,3],[7,2],[8,2],[8,1],[9,1]]},"map3_4":{"seed":[12],"start":[5,5],"goal":[10,11],"path":[[5,5],[5,6],[5,7],[5,8],[5,9],[5,10],[5,11],[6,11],[7,11],[8,11],[9,11],[10,11]]},"map4_0":{"seed":[4],"start":[10,4],"goal":[3,3],"path":[[10,4],[9,4],[9,5],[8,5],[8,6],[8,7],[7,7],[6,7],[6,6],[6,5],[5,5],[5,4],[5,3],[4,3],[3,3]]},"map4_1":{"seed":[4],"start":[7,7],"goal":[1,12],"path":[[7,7],[6,7],[5,7],[4,7],[3,7],[3,8],[3,9],[3,10],[3,11],[2,11],[1,11],[1,12]]},"map4_2":{"seed":[4],"start":[4,1],"goal":[11,11],"path":[[4,1],[3,1],[3,2],[3,3],[4,3],[5,3],[5,4],[5,5],[6,5],[6,6],[6,7],[6,8],[6,9],[5,9],[5,10],[5,11],[6,11],[7,11],[7,10],[8,10],[9,10],[10,10],[10,11],[11,11]]},"map4_3":{"seed":[6],"start":[1,8],"goal":[13,8],"path":[[1,8],[1,7],[2,7],[3,7],[4,7],[5,7],[6,7],[7,7],[8,7],[8,6],[8,5],[9,5],[9,4],[10,4],[11,4],[12,4],[12,5],[12,6],[13,6],[13,7],[13,8]]},"map4_4":{"seed":[8],"start":[5,10],"goal":[11,4],"path":[[5,10],[5,9],[6,9],[6,8],[6,7],[7,7],[8,7],[8,6],[8,5],[9,5],[9,4],[10,4],[11,4]]},"map5_0":{"seed":[0],"start":[10,4],"goal":[2,9],"path":[[10,4],[11,4],[11,5],[11,6],[11,7],[10,7],[9,7],[8,7],[8,6],[7,6],[7,5],[6,5],[5,5],[4,5],[3,5],[2,5],[1,5],[1,6],[1,7],[1,8],[2,8],[2,9]]},"map5_1":{"seed":[0],"start":[6,8],"goal":[4,13],"path":[[6,8],[7,8],[7,9],[8,9],[8,10],[8,11],[8,12],[8,13],[7,13],[6,13],[5,13],[4,13]]},"map5_2":{"seed":[2],"start":[10,7],"goal":[10,1],"path":[[10,7],[11,7],[11,6],[11,5],[11,4],[10,4],[9,4],[8,4],[8,3],[8,2],[9,2],[10,2],[10,1]]},"map5_3":{"seed":[4],"start":[1,6],"goal":[12,15],"path":[[1,6],[1,7],[1,8],[2,8],[2,9],[3,9],[3,10],[4,10],[4,11],[5,11],[6,11],[6,12],[6,13],[7,13],[8,13],[9,13],[10,13],[11,13],[12,13],[12,14],[12,15]]},"map5_4":{"seed":[5],"start":[3,10],"goal":[15,9],"path":[[3,10],[4,10],[4,11],[5,11],[6,11],[6,12],[6,13],[7,13],[8,13],[9,13],[10,13],[11,13],[12,13],[12,12],[12,11],[13,11],[14,11],[15,11],[15,10],[15,9]]}}

###########
index = 'map3_3'
###########

seed = goals[index]['seed'][0]
start = goals[index]['start']
end = goals[index]['goal']

env = DuckietownEnv(
    domain_rand=False,
    max_steps=5000,
    map_name=index,
    seed=seed,
    user_tile_start=start,
    goal_tile=end,
    randomize_maps_on_reset=False
)

env.render()


map_img, goal, start_pos = env.get_task_info()
print("start tile:", start_pos, " goal tile:", goal)
map_img = cv2.resize(map_img, None, fx=0.5,fy=0.5)
curve_angles = {(1, -1):1, (1, 1):0, (-1,1):3, (-1,-1):2}
# planner = MotionPlanner(env)
# path =planner.astar()
path = goals[index]['path']
print(path)
total_reward = 0
dts = np.array([], np.int32)

rewardList = []
predicted_pos = start_pos
trig_target = None

obs = []

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
        steering = angle_from_straight_in_rads * 5
        print("Lane alignment")
    elif lane_alignment and not direction_alignment:
        if c != 0:
            speed = 0
            steering = 6
            print("Tuning distance_to_road_center")
        elif c == 0:
            direction_alignment = True
            print("Done Direction")
    elif lane_alignment and direction_alignment and not opposite_aligment:
        print("distance :", distance_to_road_center)
        if distance_to_road_center < 0.3:
            speed = 1
            steering = -5.6
            print("Turn for correct lane")
        elif c != 0:
            direction_alignment = False
        else:
            print("Done Calibration, distance:", distance_to_road_center)
            opposite_aligment = True
            break
    else:
        break
    actions.append([speed, steering])
    obs, reward, done, info = env.step([speed, steering])
    rewardList.append(reward)
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
    speed = hand_speed
    steering = hand_steering
    if predicted_pos != trig_target:
        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        k_p = 100
        k_d = 50
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
        speed = hand_speed
        steering = hand_steering
        
    try:
        path_index = path.index(predicted_pos)
        if len(path) > path_index + 2:
            mid = path[path_index + 1]
            next_point = path[path_index + 2]
            if next_point[0] != predicted_pos[0] \
                and next_point[1] != predicted_pos[1]:
                trig_target = mid
                print("I am  controling")
                hand_speed, hand_steering = get_vw(trig_target, next_point)
    except:
        pass
    print(predicted_pos, trig_target)
    actions.append([speed, steering])
    obs, reward, done, info = env.step([speed, steering])
    rewardList.append(reward)
    total_reward += reward
    d = [int(env.cur_pos[0] * 50), int(env.cur_pos[2] * 50)]
    dts = np.append(dts,d)
    dts = dts.reshape((-1,1,2))
    map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=3)
    cv2.imshow("map", map_img)
    cv2.waitKey(10)
    env.render()
    print(reward)
    if done or (goal == [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]):
        break

count = 0
cam_angle = env.unwrapped.cam_angle
cam_angle[0] = 10

while True:
    time.sleep(1)
    move = False
    im = Image.fromarray(obs)
    prediction = predict(im) # change
    count += 1
    im.save('../result/images/%s.png' % (count))
    print(count, prediction)
    if prediction == 2:
        speed = 1
        steering = 0
        move = True

    elif prediction == 1:
        speed = 1
        steering = 8    
        move = True

    elif prediction == 3:
        speed = 1
        steering = -8
        move = True

    elif prediction == 4:
        cam_angle = env.unwrapped.cam_angle 
        cam_angle[0] -= 5

    elif prediction == 0:
        break

    if move == True:
        actions.append([speed, steering])
        obs, reward, done, info = env.step([speed, steering])
        rewardList.append(reward)
        total_reward += reward
        d = [int(env.cur_pos[0] * 50), int(env.cur_pos[2] * 50)]
        dts = np.append(dts,d)
        dts = dts.reshape((-1,1,2))
        map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=3)
        cv2.imshow("map", map_img)
        cv2.waitKey(10)
        print(reward)

    env.render()
    # INTENTION_MAPPING = {'front': 2, 'left': 1, 'right': 3,'up': 4, 'stop': 0 }

    


predicted_pos = [math.floor(env.cur_pos[0]), math.floor(env.cur_pos[2])]
np.savetxt(f'../result/{index}_seed{seed}_start_{start[0]},{start[1]}_goal_{end[0]},{end[1]}.txt',
           actions, delimiter=',')
np.savetxt(f'../result/{index}_seed{seed}_start_{start[0]},{start[1]}_goal_{end[0]},{end[1]}_reward.txt',
           rewardList, delimiter=',')
print("done", predicted_pos, total_reward, "initial", initial_reward)
pyglet.app.run()