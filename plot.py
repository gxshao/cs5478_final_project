import matplotlib.pyplot as plt
plt.plot(7.0640255488524195,7.576037345525467, 'o')
plt.plot(6.408734543655439,7.8187963560664855, 'o')
plt.plot(5.73162972704447,7.775060477799597, 'o')
plt.plot(5.311495086731902,7.242265354196745, 'o')
plt.plot(5.426859980746914,6.573628905179452, 'o')
plt.plot(5.73006896256654,5.946547438908451, 'o')
plt.plot(5.84849828219151,5.260150069850375, 'o')
plt.plot(5.678900294080709,4.591578533323135, 'o')
plt.plot(5.247605661273089,4.04463082859441, 'o')
plt.plot(4.637014857459064,3.723803386932428, 'o')
plt.plot(3.941924176444903,3.6789057571068793, 'o')
plt.plot(3.2468334954307405,3.634008127281338, 'o')
plt.plot(2.5517428144165786,3.5891104974557893, 'o')
plt.plot(1.9129853848691778,3.360259659739283, 'o')
plt.plot(1.6561551818594467,2.7322294004782104, 'o')
plt.plot(1.486557193748657,2.063657863950965, 'o')
plt.plot(1.2229048369266073,1.4189453026192793, 'o')
plt.show()


# Test
# pos = env.cur_pos
# angle = env.cur_angle
# print("initial pose:", pos, angle)
# print("predcit:", motion_predict(pos, angle, [0.1, 1.57]))
# for i in range(30):
#     env.step([0.1, 1.57])
#     env.render()
# print("ref pose:",env.cur_pos, env.cur_angle)



# path = [[7, 7], [6, 7], [5, 7], [5, 6], [5, 5], [4, 5], [3, 5], [3, 4], [3, 3], [2, 3], [1, 3], [1, 2], [1, 1]]

# def closest_curve_point(pos, curves, angle=None):
#     curve_headings = curves[:, -1, :] - curves[:, 0, :]
#     #curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
    
#     dir_vec = env.get_dir_vec(angle)
#     dot_prods = np.dot(curve_headings, dir_vec)

#     cps = curves[np.argmax(dot_prods)]
#     t = bezier_closest(cps, pos)
#     point = bezier_point(cps, t)
#     tangent = bezier_tangent(cps, t)
#     return point, tangent

# def get_lane_pos2(pos, angle, target_curve):
#     point, tangent = closest_curve_point(pos, target_curve, angle)
#     if point is None:
#         msg = 'Point not in lane: %s' % pos
#         raise Exception(msg)
#     assert point is not None
#     dirVec = env.get_dir_vec(angle)
#     dotDir = np.dot(dirVec, tangent)
#     dotDir = max(-1, min(1, dotDir))
#     posVec = pos - point
#     upVec = np.array([0, 1, 0])
#     rightVec = np.cross(tangent, upVec)
#     signedDist = np.dot(posVec, rightVec)
#     angle_rad = math.acos(dotDir)
#     if np.dot(dirVec, rightVec) < 0:
#         angle_rad *= -1
#     angle_deg = np.rad2deg(angle_rad)
#     return LanePosition(dist=signedDist, dot_dir=dotDir, angle_deg=angle_deg,
#                         angle_rad=angle_rad)
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

# def get_curve(curve, offset, map_img):
#     tile = env._get_tile(offset[0], offset[1])
#     print("angle match:", current_key, tile['angle'])
#     diff = current_key - tile['angle']
#     # if diff > 0:
#         #clockwise
#         # curve = np.rot90(curve, diff)
#     # elif diff < 0:
#         # anti
#         # curve = np.rot90(curve, 4 + diff)
         
#     curve *= env.road_tile_size
#     mat = gen_rot_matrix(np.array([0, 1, 0]), (tile['angle']) * math.pi / 2)
#     curve = np.matmul(curve, mat)
#     curve += np.array([(offset[0] + 0.5) * env.road_tile_size, 0, (offset[1] + 0.5) * env.road_tile_size])

#     for c in curve:
#         dts = np.array([], np.int32)
#         for p in c:
#             dts = np.append(dts, [int((p[0]) * 100), int((p[2]) * 100)])
#         dts = dts.reshape((-1,1,2))
#         map_img = cv2.polylines(map_img,[dts],False,(0,0,255), thickness=3)
#         cv2.imshow("map", map_img)
#         cv2.waitKey(100)
        
        
#     return curve

# curve_list = {(1,1):np.array([
#                 [
#                     [-0.30, 0, -0.20],
#                     [-0.30, 0, 0.20],
#                     [-0.20, 0, 0.30],
#                     [0.20, 0, 0.30],
#                 ],
#             ]),
#             (-1,-1):np.array([
#                  [
#                     [-0.30, 0, -0.20],
#                     [0.20, 0, -0.20],
#                     [0.20, 0, -0.20],
#                     [0.20, 0, 0.30],
#                 ],
#             ]),
#             # E-N
#             (1,-1): np.array([
#                 [
#                     [-0.20, 0, -0.20],
#                     [0.20, 0, -0.20],
#                     [-0.20, 0, -0.20],
#                     [-0.20, 0, 0.20],
#                 ]
#             ]),
#             # W-S
#             (-1,1): np.array([
#                 [
#                      [ 0.2,  0.0  , 0.3],
#                      [ 0.2 , 0.0 , -0.2],
#                      [ 0.2 , 0.0 , -0.2],
#                      [0.3 , 0.0 , -0.2]
#                 ]
#             ])
#             }
# print(np.fliplr(curve_list[(1, -1)]))
# print(np.flipud(curve_list[(1, -1)]))
# base_curve = np.array([
#                 [
#                     [-0.20, 0, -0.20],
#                     [0.20, 0, -0.20],
#                     [-0.20, 0, -0.20],
#                     [-0.20, 0, 0.20],
#                 ],])
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
# def motion_predict(pos, angle, action):
#     x, y = pos, angle
#     for i in range(30):
#         vel, angle = action

#         # Distance between the wheels
#         baseline = env.unwrapped.wheel_dist

#         # assuming same motor constants k for both motors
#         k_r = env.k
#         k_l = env.k

#         # adjusting k by gain and trim
#         k_r_inv = (env.gain + env.trim) / k_r
#         k_l_inv = (env.gain - env.trim) / k_l

#         omega_r = (vel + 0.5 * angle * baseline) / env.radius
#         omega_l = (vel - 0.5 * angle * baseline) / env.radius

#         # conversion from motor rotation rate to duty cycle
#         u_r = omega_r * k_r_inv
#         u_l = omega_l * k_l_inv

#         # limiting output to limit, which is 1.0 for the duckiebot
#         u_r_limited = max(min(u_r, env.limit), -env.limit)
#         u_l_limited = max(min(u_l, env.limit), -env.limit)

#         vels = np.array([u_l_limited, u_r_limited])

#         x, y = _update_pos(x, y, 0.102, vels * env.robot_speed * 1, env.delta_time)
#     return x, y
