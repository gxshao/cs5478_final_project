import argparse
import math

import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv
import os
import time
import torch
from PIL import Image
from nnModel import predict

parser = argparse.ArgumentParser()

parser.add_argument('--path', '-m', default="./test/", type=str)
args = parser.parse_args()

target_folder = args.path

targets = arr = os.listdir(target_folder)
# print(targets)
test_img_results = "test_" + time.strftime("%Y%m%d-%H%M%S")
os.makedirs(test_img_results)

def create_test(filename, map, seed, start_pose, goal_pose):
    obs = []
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

    actionList = []
    count = 0
    defalt_cam_Y = 5
    cam_angle = env.unwrapped.cam_angle
    cam_angle[0] = 5
    defalt_cam_X = 0
    print("defalt_cam_dir",defalt_cam_X)
    rewardCNN= 0
    leftCount= 0
    cc= 0
    while True:
        move = False
        viewImage = Image.fromarray(env.img_array_human)
        im = viewImage.rotate(180).transpose(method=Image.FLIP_LEFT_RIGHT)
        prediction = predict(im) 
        print(-cam_angle[1], prediction)
        if cc >= 8:
            break

        if prediction == 2:
            if -cam_angle[1] < 130:
                speed = 1
                steering = 0
                move = True

            elif -cam_angle[1] >= 130 and -cam_angle[1] < 230:
                cam_angle[1] = defalt_cam_X
                cam_angle[0] = defalt_cam_Y
                break

            elif -cam_angle[1] >= 230 and -cam_angle[1] < 360:
                speed = 1
                steering = -(0.03 * cam_angle[1] + 19.5)
                move = True

            elif -cam_angle[1] >= 360 :
                cam_angle[1] = defalt_cam_X
                cam_angle[0] = defalt_cam_Y
                break

        elif prediction == 3:
            if -cam_angle[1] >= 230 and -cam_angle[1] < 360:
                speed = 1
                steering = -(0.03 * cam_angle[1] + 19.5)
                move = True
            else:
                cam_angle[1] = defalt_cam_X
                cam_angle[0] = defalt_cam_Y
                break

        elif prediction == 0:
            cam_angle[1] = defalt_cam_X
            cam_angle[0] = defalt_cam_Y
            break

        else:
            if -cam_angle[1] > 360:
                if cam_angle[0] < -10:
                    break
                else:
                    cam_angle[0] -=5
                    cam_angle[1] = defalt_cam_X
                    speed = 1
                    steering = 0
                    move = True
            else:
                cam_angle = env.unwrapped.cam_angle
                cam_angle[1] -= 10

        if move == True:
            cc +=1
            actionList.append([speed, steering])
            obs, reward, done, info = env.step([speed, steering])
            rewardCNN += reward
            d = [int(env.cur_pos[0] * 50), int(env.cur_pos[2] * 50)]
            dts = np.append(dts,d)
            dts = dts.reshape((-1,1,2))
            map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=3)
            cv2.imshow("map", map_img)
            print(reward)
        cv2.waitKey(50)
        env.render()

    time.sleep(1)
    cam_angle[0] = 5
    cam_angle[1] = 0
    env.render()
    time.sleep(1)
    print("pre-cnn finished")
    while True:
        move = False
        # im = Image.fromarray(obs)
        viewImage = Image.fromarray(env.img_array_human)
        im = viewImage.rotate(180).transpose(method=Image.FLIP_LEFT_RIGHT)
        im.save(f'./result/images/{map_name}_seed{seed}_start_{start_pose[0]},{start_pose[1]}_goal_{goal_pose[0]},{goal_pose[1]}.png')
        prediction = predict(im) # change
        print(count, prediction, leftCount)
        if leftCount >= 30:
            speed = 1
            steering = 0
            move = True
            cam_angle[0] -= 5
        else:
            if prediction == 2:
                leftCount +=1 
                speed = 1
                steering = 0
                move = True

            elif prediction == 1:
                leftCount =0
                speed = 1
                steering = 9.8    
                move = True

            elif prediction == 3:
                leftCount =0
                speed = 1
                steering = -9.8
                move = True

            elif prediction == 4:
                leftCount =0
                cam_angle = env.unwrapped.cam_angle 
                if cam_angle[0] < -15:
                    speed = 1
                    steering = 0
                    move = True
                else:
                    cam_angle[0] -= 5

            elif prediction == 0:
                break

            if move == True:
                actionList.append([speed, steering])
                obs, reward, done, info = env.step([speed, steering])
                rewardCNN += reward
                d = [int(env.cur_pos[0] * 50), int(env.cur_pos[2] * 50)]
                dts = np.append(dts,d)
                dts = dts.reshape((-1,1,2))
                map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=3)
                cv2.imshow("map", map_img)
                print(reward)

            if count > 50:
                break
        cv2.waitKey(50)
        env.render()
    print("CNN finished:", filename,'finale pos:', [round(env.cur_pos[0], 2), round(env.cur_pos[2], 2)], "cnn reward:", rewardCNN, "total reward",total_reward+rewardCNN )
    # INTENTION_MAPPING = {'front': 2, 'left': 1, 'right': 3,'up': 4, 'stop': 0 }
    np.savetxt(f'./result/{map_name}_seed{seed}_start_{start_pose[0]},{start_pose[1]}_goal_{goal_pose[0]},{goal_pose[1]}_action.txt',
           actionList, delimiter=',')
    np.savetxt(f'./result/reward/{map_name}_seed{seed}_start_{start_pose[0]},{start_pose[1]}_goal_{goal_pose[0]},{goal_pose[1]}_reward.txt',
           [total_reward,rewardCNN,total_reward+rewardCNN], delimiter=',')
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