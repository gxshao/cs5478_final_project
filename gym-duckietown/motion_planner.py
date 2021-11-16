import numpy as np
import math
import cv2
import json 
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
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.g = 0
        self.h = 0
        self.prev = None

    def f_cost(self):
        return self.g + self.h

    def equals(self, p):
        return p.x == self.x and p.y == self.y
    
    def __eq__(self, other):
        return self.equals(other)

    def __hash__(self):
        return hash((self.x, self.y))
    
    def __str__(self) -> str:
        return '[' + str(self.x) + ',' + str(self.y) + ']'
    
    def __repr__(self) -> str:
        return '[' + str(self.x) + ',' + str(self.y) + ']'
class MotionPlanner:
    # Heuristic and also G value
    def cal_cost(self, current_point, target_point):
        dstX = np.abs(current_point.x - target_point.x)
        dstY = np.abs(current_point.y - target_point.y)

        if dstX > dstY:
            return 14 * dstY + 10 * (dstX - dstY)
        return 14 * dstX + 10 * (dstY - dstX)
        
    def get_neighbours(self, width, height, current_point):
        results = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                # Current point
                if i == 0 and j == 0:
                    continue
                if i == j or (i == 1 and j == -1) or (i == -1 and j == 1):
                   continue
                x = current_point.x + i
                y = current_point.y + j
                if x == 0 or y == 0:
                    continue
                if 0 <= x < width and 0 <= y < height:
                    n_p = Point(x, y)
                    if i == 0:
                        if j < 0:
                            n_p.angle = np.pi * 1.5
                        elif j > 0:
                            n_p.angle = np.pi / 2
                    elif j == 0:
                        if i > 0:
                            n_p.angle = 0
                        elif i < 0:
                            n_p.angle = np.pi
                    results.append(n_p)
        return results
    
    def __init__(self, env):
        self.env = env
        self.map_img = None
        self.map_width = 0
        self.map_height = 0
    
    def is_drivable(self, point)-> bool:
        obstacle = [0, 0, 0]
        mapdata = self.map_img.tolist()
        return obstacle != mapdata[point.y * 100][point.x * 100]

    def astar(self):
        map_img, goal, start_pos = self.env.get_task_info()
        start = Point(start_pos[0], start_pos[1])
        start.angle = self.env.cur_angle
        end = Point(goal[0], goal[1])
        print(start.x, start.y, end.x, end.y)
        self.map_img = map_img
        self.map_width = self.env.grid_width
        self.map_height = self.env.grid_height
        open_list = [start]
        path = []
        close_set = set()
        dts = np.array([], np.int32)
        while len(open_list) > 0:
            current_point = open_list[0]
            # priority was implemented manually...
            for i in range(1, len(open_list)):
                if (current_point.f_cost() > open_list[i].f_cost() or current_point.f_cost() == open_list[i].f_cost) \
                        and open_list[i].h < current_point.h:
                    current_point = open_list[i]

            open_list.remove(current_point)
            close_set.add(current_point)

            if current_point.equals(end):
                head = current_point
                while head != start:
                    path.append([head.x, head.y])
                    d = [int(head.x * 100), int(head.y * 100)]
                    dts = np.append(dts, d)
                    head = head.prev
                path.append([start.x, start.y])
                path.reverse()
                break

            neighbours = self.get_neighbours(int(self.map_width) , int(self.map_height), current_point)
            print(neighbours)
            for neighbour in neighbours:
                ## Find v and w for neighbour 
                if neighbour in close_set:
                    close_set.add(neighbour)
                    continue
                if not self.is_drivable(neighbour):
                    continue
                n_cost = current_point.g + self.cal_cost(current_point, neighbour)
                if n_cost < neighbour.g or neighbour not in open_list:
                    neighbour.g = n_cost
                    neighbour.h = self.cal_cost(neighbour, end)
                    neighbour.prev = current_point
                    # self.map_img = cv2.circle(self.map_img, (current_point.x * 100,current_point.y * 100), radius=0, color=(0, 0, 255), thickness=10)
                    # cv2.imshow("map", self.map_img)
                    # cv2.waitKey(10)
                    print("current:",current_point.x, current_point.y, current_point.angle, "selected:", neighbour.x, neighbour.y, "Cost", neighbour.g)
                    open_list.append(neighbour)
        
        
        dts = dts.reshape((-1,1,2))
        # self.map_img = cv2.polylines(self.map_img,[dts],False,(0,0,255), thickness=30)
        # cv2.imshow("map", self.map_img)
        # cv2.waitKey(10)
        return path

# goal = "{\"map1_0\": { 		\"seed\": [1], 		\"start\": [0, 1], 		\"goal\": [5, 1] 	}, 	\"map1_1\": { 		\"seed\": [0], 		\"start\": [0, 1], 		\"goal\": [70, 1] 	}, 	\"map1_2\": { 		\"seed\": [2], 		\"start\": [2, 1], 		\"goal\": [21, 1] 	}, 	\"map1_3\": { 		\"seed\": [6], 		\"start\": [5, 1], 		\"goal\": [65, 1] 	}, 	\"map1_4\": { 		\"seed\": [5], 		\"start\": [50, 1], 		\"goal\": [90, 1] 	}, 	\"map2_0\": { 		\"seed\": [1], 		\"start\": [7, 7], 		\"goal\": [1, 1] 	}, 	\"map2_1\": { 		\"seed\": [2], 		\"start\": [3, 6], 		\"goal\": [7, 1] 	}, 	\"map2_2\": { 		\"seed\": [5], 		\"start\": [1, 6], 		\"goal\": [3, 4] 	}, 	\"map2_3\": { 		\"seed\": [4], 		\"start\": [1, 2], 		\"goal\": [5, 4] 	}, 	\"map2_4\": { 		\"seed\": [4], 		\"start\": [7, 4], 		\"goal\": [4, 7] 	}, 	\"map3_0\": { 		\"seed\": [1], 		\"start\": [5, 7], 		\"goal\": [2, 2] 	}, 	\"map3_1\": { 		\"seed\": [2], 		\"start\": [5, 11], 		\"goal\": [1, 7] 	}, 	\"map3_2\": { 		\"seed\": [3], 		\"start\": [10, 5], 		\"goal\": [7, 11] 	}, 	\"map3_3\": { 		\"seed\": [4], 		\"start\": [2, 4], 		\"goal\": [9, 1] 	}, 	\"map3_4\": { 		\"seed\": [12], 		\"start\": [5, 5], 		\"goal\": [10, 11] 	}, 	\"map4_0\": { 		\"seed\": [4], 		\"start\": [10, 4], 		\"goal\": [3, 3] 	}, 	\"map4_1\": { 		\"seed\": [4], 		\"start\": [7, 7], 		\"goal\": [1, 12] 	}, 	\"map4_2\": { 		\"seed\": [4], 		\"start\": [4, 1], 		\"goal\": [11, 11] 	}, 	\"map4_3\": { 		\"seed\": [6], 		\"start\": [1, 8], 		\"goal\": [13, 8] 	}, 	\"map4_4\": { 		\"seed\": [8], 		\"start\": [5, 10], 		\"goal\": [11, 4] 	}, 	\"map5_0\": { 		\"seed\": [0], 		\"start\": [10, 4], 		\"goal\": [2, 9] 	}, 	\"map5_1\": { 		\"seed\": [0], 		\"start\": [6, 8], 		\"goal\": [4, 13] 	}, 	\"map5_2\": { 		\"seed\": [2], 		\"start\": [10, 7], 		\"goal\": [10, 1] 	}, 	\"map5_3\": { 		\"seed\": [4], 		\"start\": [1, 6], 		\"goal\": [12, 15] 	}, 	\"map5_4\": { 		\"seed\": [5], 		\"start\": [3, 10], 		\"goal\": [15, 9] 	} }"
# # print(goal)

# goal = json.loads(goal)

# result = {}
# for key in goal:
#     map_name = key
#     seed = goal[key]['seed'][0]
#     start = goal[key]['start']
#     end = goal[key]['goal']
#     print(key, seed, start,end)
#     env = DuckietownEnv(
#         domain_rand=False,
#         max_steps=5000,
#         map_name=map_name,
#         seed=seed,
#         user_tile_start=start,
#         goal_tile=end,
#         randomize_maps_on_reset=False   
#         )
#     planner = MotionPlanner(env)
#     path =planner.astar()
#     goal[key]['path'] = path
#     print(key,"  ---  " ,path)

# print(json.dumps(goal))

# with open('new_goal.json', 'w') as f:
#     f.write(json.dumps(goal))