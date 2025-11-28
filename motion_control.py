import asyncio
import numpy as np
from tdmclient import ClientAsync

from local_avoidance import *

FORWARD_SPEED = 100
GAIN = 2
GAIN_ANGLE = 60

KIDNAPPING_THR = 40

STATE = "UNKOWN"

INDEX = 0

robot_angle = 0 # temp, has to be replace with the v data

def motors(l, r):
    return {"motor.left.target": [int(l)], "motor.right.target": [int(r)]}

def update_state(node):
    global STATE
    prox_h = list(node["prox.horizontal"])
    prox_gnd = list(node["prox.ground.delta"])
    # print(f'Prox = {prox_h}')
    # print(f'Prox gnd = {prox_gnd}')


    if test_kidnapping(prox_gnd, STATE):
        STATE = "KIDNAPPED"
        #print(f'State updated to {STATE}.')
        return
    elif test_no_obstacle(prox_h) and STATE != "MOVE":
        STATE = "MOVE"
        #print(f'State updated to {STATE}.')
    elif test_obstacle_detected(prox_h) and STATE != "OBSTACLE":
        STATE = "OBSTACLE"
        #print(f'State updated to {STATE}.')
    # else:
    #     print("No state change.")
    return

def test_kidnapping(prox, prev_state):
    global KIDNAPPING_THR
    return max(prox) < KIDNAPPING_THR

async def follow_instruction(client, node, path, v):
    #await gradient_following(client, node) # change for the instruction to follow
    await path_following(client, node, path, v)
    return

async def gradient_following(client, node): # instruction for testing
    global GAIN, FORWARD_SPEED
    prox_gnd = list(node["prox.ground.delta"])
    grad = prox_gnd[1] - prox_gnd[0]
    print(f'prox = {prox_gnd} ; grad = {grad}')
    left_speed = FORWARD_SPEED - grad * GAIN
    right_speed = FORWARD_SPEED + grad * GAIN
    await node.set_variables(motors(left_speed, right_speed))
    await client.sleep(0.25)
    return

async def path_following(client, node, path, v):
    global INDEX
    #path = [(20, 0), (10, np.pi/2), (10, np.pi/2), (14.14, np.pi/4), (0, 3*np.pi/4)]
    path = [(10, 0), (10, np.pi/2), (10, np.pi), (10, -np.pi/2)]
    while INDEX < len(path) and not test_obstacle_detected(list(node["prox.horizontal"])):
        await angle_correction(client, node, path[INDEX][1], v)

        dist_count = 0
        while dist_count < path[INDEX][0] and not test_obstacle_detected(list(node["prox.horizontal"])):
            await move_to(client, node, 1)
            print(f"move dist = {dist_count}")
            dist_count += 1

        INDEX += 1
    return

async def angle_correction(client, node, target_angle, v):
    global GAIN_ANGLE
    # rot_speed = 4.5 / np.pi
    # if target_angle > 0:
    #     await node.set_variables(motors(-100, 100))
    #     await client.sleep(target_angle * rot_speed)
    # elif target_angle < 0:
    #     await node.set_variables(motors(100, -100))
    #     await client.sleep(-target_angle * rot_speed)

    epsilon = 0.1
    _, _, robot_angle = v.get_thymio_pos(v.get_image(v._Vision__cap, False))
    error_angle = (target_angle - robot_angle)
    if error_angle > np.pi:
        error_angle = error_angle - 2 * np.pi
    if error_angle < -np.pi:
        error_angle = error_angle + 2 * np.pi

    while abs(error_angle) > epsilon:
        left_speed = error_angle * GAIN_ANGLE 
        right_speed = - error_angle * GAIN_ANGLE
        print(f"angle target = {target_angle}; angle robot = {robot_angle}; error = {error_angle}")
        await node.set_variables(motors(left_speed, right_speed))
        _, _, robot_angle = v.get_thymio_pos(v.get_image(v._Vision__cap, False))
        error_angle = (target_angle - robot_angle)
        if error_angle > np.pi:
            error_angle = error_angle - 2 * np.pi
        if error_angle < -np.pi:
            error_angle = error_angle + 2 * np.pi
        await client.sleep(0.2)
    await node.set_variables(motors(0, 0))
    print("Robot aligned to target!")
    await client.sleep(1)

async def move_to(client, node, dist):
    speed = 1 / 3.5

    await node.set_variables(motors(100, 100))
    await client.sleep(dist * speed)
    #await node.set_variables(motors(0, 0))

    return

async def motion_control(client, node, path, v):
    update_state(node)
    if STATE == "KIDNAPPED":
        await node.set_variables(motors(0, 0))
        return True
        print("Kidnapped")
    elif STATE == "OBSTACLE":
        await avoid_obstacle(client, node)
        return True
    elif STATE == "MOVE":
        await follow_instruction(client, node, path, v)
        return False
    else:
        raise("State error")