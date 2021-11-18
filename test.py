import argparse
import math

import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv
import os
import time

parser = argparse.ArgumentParser()

parser.add_argument('--path', '-m', default="./test/", type=str)
args = parser.parse_args()

target_folder = args.path

targets = arr = os.listdir(target_folder)
# print(targets)
test_img_results = "test_" + time.strftime("%Y%m%d-%H%M%S")
os.makedirs(test_img_results)

def create_test(filename, map, seed, start_pose, goal_pose):
    total_reward = 0
    print("Test start:", filename,'initial pos:',start_pose)
    env = DuckietownEnv(
        domain_rand=False,
        max_steps=5000,
        map_name=map,
        seed=seed,
        user_tile_start=start_pose,
        goal_tile=goal_pose,
        randomize_maps_on_reset=False
    )
    env.render()
    map_img, goal, start_pos = env.get_task_info()
    map_img = cv2.resize(map_img, None, fx=0.5, fy=0.5)
    print("start tile:", start_pos, " goal tile:", goal)
    
    actions = np.loadtxt(target_folder + filename, delimiter=',')
    dts = np.array([], np.int32)
    for (speed, steering) in actions:

        obs, reward, done, info = env.step([speed, steering])
        total_reward += reward
        d = [int(env.cur_pos[0] * 50), int(env.cur_pos[2] * 50)]
        dts = np.append(dts,d)
        dts = dts.reshape((-1,1,2))
        map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=3)
        cv2.imshow("map", map_img)
        cv2.waitKey(50)
        env.render()
    print("Test finished:", filename,'finale pos:', [round(env.cur_pos[0], 2), round(env.cur_pos[2], 2)], "final reward:", total_reward)
    cv2.imwrite(test_img_results + '/'+ map + '.jpg', map_img)
    time.sleep(2)
    env.window.close()
    cv2.destroyAllWindows()

count = 1
for file in targets:
    filename_arr = file.split('_')
    map_name = filename_arr[0] + '_' + filename_arr[1]
    seed = filename_arr[2].replace('seed', '')
    start_pose = filename_arr[4].split(',')
    goal_pose = filename_arr[6].split(',')
    print('\n\n')
    print ("current file count:", count, "Target files left:", len(targets) - count)
    create_test(file,map_name, int(seed), [int(start_pose[0]), int(start_pose[1])], [int(goal_pose[0]), int(goal_pose[1].split('.')[0])])
    count += 1