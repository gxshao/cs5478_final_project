import argparse
from os import curdir, strerror
from hybrid_planner import *
import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import _update_pos
import pyglet
import time
# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map2_0", type=str)
parser.add_argument('--seed', '-s', default=2, type=int)
parser.add_argument('--start-tile', '-st', default="7,7", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="1,1", type=str, help="two numbers separated by a comma")
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




planner = HybridAStarPlanner(env)
path, dots =planner.astar()
for p in path:
    for i in range(0, 30):
        env.step([p[0][i], p[1][i]])
        env.render()
    print("current pose:",env.cur_pos)
print(dots)
print("done")

# cv2.imshow("map", map_img)
# cv2.waitKey(200)

pyglet.app.run()