#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose. 
"""

import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
import cv2 

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='map2_0')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = False,
        draw_bbox = False,
        seed=1,
        user_tile_start=[7, 7],
        goal_tile=[1,1],
    )
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

total_reward = 0
map_img,_,_ = env.get_task_info()
dts = np.array([], np.int32)
c = False
while True:
    time.sleep(0.01)
    lane_pose = None
    try:
        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    except:
        continue
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad
    print("Reward function: ", distance_to_road_center, angle_from_straight_in_rads)

    ###### Start changing the code here.
    # TODO: Decide how to calculate the speed and direction.

    # k_p = 80
    # k_d = 10
    
    # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)
    
    #speed = 1 # TODO: You should overwrite this value
    
    k_p = 10
    k_d = 2
    
    # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)
    
    speed = 0.3 # TODO: You should overwrite this value
    k_i = 0.5
    steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads #+ k_i * lane_pose.dot_dir# TODO: You should overwrite this value

    ###### No need to edit code below.        

    obs, reward, done, info = env.step([speed, steering])
    total_reward += reward
    d = [int(env.cur_pos[0] * 100), int(env.cur_pos[2] * 100)]
    dts = np.append(dts,d)
    #print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))
    
    dts = dts.reshape((-1,1,2))
    map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=10)
    cv2.imshow("map", map_img)
    cv2.waitKey(10)
    env.render()

    # if done:
    #     if reward < 0:
    #         print('*** CRASHED ***')
    #     print ('Final Reward = %.3f' % total_reward)
    #     break
