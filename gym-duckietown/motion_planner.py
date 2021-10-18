import numpy as np
import math

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
    
    def __init__(self, map_pixel, map_width, map_height, start, goal, robot_size=1) -> None:
        self.map_data = map_pixel.tolist()
        self.start = Point(start[0], start[1])
        self.end = Point(goal[0], goal[1])
        self.map_width = map_width
        self.map_height = map_height
        self.resolution = 100
        
        self.robot_size = robot_size
        
        # We cannot navigate to the resturant directly, so go to it's neighbour
        if not self.is_drivable(self.end):
            neighbours = self.get_neighbours(self.map_width, self.map_height, self.end)
            for neighbour in neighbours:
                if self.is_drivable(neighbour):
                    self.end = neighbour
        
        print("start:", start)
        print("goal :", goal)
    
    def is_drivable(self, point):
        obstacle = [0, 0, 0]
        return obstacle != self.map_data[point.y * self.resolution][point.x * self.resolution]
    
    def astar(self):

        open_list = [self.start]
        path = []
        close_set = set()

        while len(open_list) > 0:
            current_point = open_list[0]
            # priority was implemented manually...
            for i in range(1, len(open_list)):
                if (current_point.f_cost() > open_list[i].f_cost() or current_point.f_cost() == open_list[i].f_cost) \
                        and open_list[i].h < current_point.h:
                    current_point = open_list[i]

            open_list.remove(current_point)
            close_set.add(current_point)

            if current_point.equals(self.end):
                head = current_point
                while head != self.start:
                    path.append([head.x, head.y])
                    head = head.prev
                path.append([self.start.x, self.start.y])
                path.reverse()
                break

            neighbours = self.get_neighbours(self.map_width, self.map_height, current_point)
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
                    neighbour.h = self.cal_cost(neighbour, self.end)
                    neighbour.prev = current_point
                    open_list.append(neighbour)
        return path
