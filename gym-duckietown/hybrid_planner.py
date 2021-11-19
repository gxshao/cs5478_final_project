from os import access, environ
from types import CodeType
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import _update_pos
import math
import cv2

# Eculidean distance for basic Hybrid A*
def cal_cost(current_point, target_point):
    dstX = np.abs(current_point.x - target_point.x)
    dstY = np.abs(current_point.y - target_point.y)
    return np.sqrt(dstY * dstY + dstX * dstX)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.g = 0
        self.h = 0
        self.v = 0
        self.w = 0
        self.prev = None

    def f_cost(self):
        return self.g + self.h

    def equals(self, p):
        return p.x == self.x and p.y == self.y

    # Round equal in order to discretize the state space
    def round_equal(self, p):
        return round(p.x) == round(self.x) and round(p.y) == round(self.y)
    
    def close(self, p):
        return cal_cost(self, p) < 0.5
    
    def __eq__(self, other):
        return self.round_equal(other)
    # Same here
    def __hash__(self):
        return hash((round(self.x, 1), round(self.y, 1)))
    
    def __str__(self) -> str:
        return str([self.x, self.y, self.angle]) 
    def __repr__(self) -> str:
        return str([self.x, self.y, self.angle])
    
ITERATION = 100000
def motion_predict(pos, face, action, env, map_img):
    x, y = pos, face
    h = 0
    result_action = [[],[]]
    pts = np.array([], np.int32)
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
        
        d = [int(x[0] * 100), int(x[2] * 100)]
        pts = np.append(pts,d)

        x, y = _update_pos(x, y, 0.102, vels * env.robot_speed * 1, env.delta_time)
        if not env._valid_pose(x, y):
            return None
        result_action[0].append(vel)
        result_action[1].append(angle)
        h += env.compute_reward(x, vel, angle)
    
    pts = pts.reshape((-1,1,2))
    cv2.polylines(map_img,[pts],False,(0,0,255))
    cv2.imshow("map", map_img)
    cv2.waitKey(10)
    return x, y, h, result_action

def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

class HybridAStarPlanner:
    
    def __init__(self, env) -> None:
        self.env = env
        self.map_img = None

    # Value G calculation
    def arc_length(self, current_point, target_point):
        time_w = 0.5
        theta = target_point.angle - current_point.angle
        w = np.abs(theta) / time_w
        return w * cal_cost(current_point, target_point) * time_w 

    def get_neighbours(self, current_point):

        results = set()
        candidates = []
        # print("get_neighbours:", current_point)
        # Discretize the action space by tuning the set
        for j in ( -0.6, -0.3,  0, 0.3, 0.6):
            candidates.append([0.5, j])

        # It's a important value for algorithm convergence
        extra_h = []
        for p in candidates:
            prediation_result = motion_predict([current_point.x, 0, current_point.y], current_point.angle, p, self.env, self.map_img)
            if prediation_result is None:
                continue
            pos, angle, h, action_result = prediation_result
            neighbour = Point(pos[0], pos[2])
            neighbour.angle = angle
            neighbour.v = action_result[0]
            neighbour.w = action_result[1]
            results.add(neighbour)
            extra_h.append(h)

        return results, extra_h

    def astar(self):
        map_img, goal, start_pos = self.env.get_task_info()
        start = Point(self.env.cur_pos[0], self.env.cur_pos[2])
        start.angle = self.env.cur_angle
        end = Point(goal[0], goal[1])
        print(start.x, start.y, end.x, end.y)
        self.map_img = map_img
        #self.map_img = zoom(map_img,0.5)
        
        open_list = [start]
        path = []
        dots = []
        close_set = set()
        ps = []
        iteration = 0
        while len(open_list) > 0 and iteration < ITERATION:
            iteration += 1
            current_point = open_list[0]
            # priority implemented manually...
            for i in range(1, len(open_list)):
                if (current_point.f_cost() > open_list[i].f_cost()
                    or current_point.f_cost() == open_list[i].f_cost) \
                        and open_list[i].h < current_point.h:
                    current_point = open_list[i]
           
            open_list.remove(current_point)
            close_set.add(current_point)
      
            if current_point.close(end):
                head = current_point
                while head != start:
                    ps.append(head) 
                    head = head.prev
                    dots.append([head.x, head.y])
                    if head.v == 0 and head.w == 0:
                        continue
                    path.append([head.v, head.w])
                path.reverse()
                dots.reverse()
                ps.reverse()
                for point in ps:
                    print("plt.plot(" + str(point.x) + "," + str(point.y) + ", \'o\')")
                break

            neighbours,extra_h = self.get_neighbours(current_point)
            # Add extra H cost to the point because it might to hit the wall
            # current_point.h += extra_h
            i = 0
            for neighbour in neighbours:
                if neighbour in close_set:   
                    continue
          
                n_cost = current_point.g + self.arc_length(current_point, neighbour)
                if n_cost < neighbour.g or neighbour not in open_list:
                    neighbour.g = n_cost
                    neighbour.h = cal_cost(neighbour, end)  + extra_h[i] * 0.5
                    
                    neighbour.prev = current_point
                    print("current:",current_point.x, current_point.y, current_point.angle, "selected:", neighbour.x, neighbour.y, "Cost", neighbour.g,"extra", extra_h[i])
                    
                    open_list.append(neighbour)
                i += 1
        return path, dots 