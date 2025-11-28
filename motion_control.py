import asyncio
import numpy as np
from tdmclient import ClientAsync

from local_avoidance import *
from vision import get_thymio_pos

FORWARD_SPEED = 100
GAIN = 2
GAIN_AVD = 5
GAIN_ANGLE = 5

KIDNAPPING_THR = 40

state = "UNKOWN"

INDEX = 0

robot_angle = 0 # temp, has to be replace with the vision data

def motors(l, r):
    return {"motor.left.target": [int(l)], "motor.right.target": [int(r)]}

def update_state(node):
    prox_h = list(node["prox.horizontal"])
    prox_gnd = list(node["prox.ground.delta"])
    # print(f'Prox = {prox_h}')
    # print(f'Prox gnd = {prox_gnd}')


    if test_kidnapping(prox_gnd, state):
        state = "KIDNAPPED"
        print(f'State updated to {state}.')
        return
    elif test_no_obstacle(prox_h) and state != "MOVE":
        state = "MOVE"
        print(f'State updated to {state}.')
    elif test_obstacle_detected(prox_h) and state != "OBSTACLE":
        state = "OBSTACLE"
        print(f'State updated to {state}.')
    else:
        print("No state change.")
    return

def test_kidnapping(prox, prev_state):
    return max(prox) < KIDNAPPING_THR

async def follow_instruction(client, node, vision):
    #await gradient_following(client, node) # change for the instruction to follow
    await path_following(client, node)
    return

async def gradient_following(client, node): # instruction for testing
    prox_gnd = list(node["prox.ground.delta"])
    grad = prox_gnd[1] - prox_gnd[0]
    print(f'prox = {prox_gnd} ; grad = {grad}')
    left_speed = FORWARD_SPEED - grad * GAIN
    right_speed = FORWARD_SPEED + grad * GAIN
    await node.set_variables(motors(left_speed, right_speed))
    await client.sleep(0.25)
    return

async def path_following(client, node, vision):
    path = [(20, 0), (10, np.pi/2), (10, np.pi/2), (14.14, np.pi/4), (0, 3*np.pi/4)]
    while INDEX < len(path) and not test_obstacle_detected(list(node["prox.horizontal"])):
        await angle_correction(client, node, path[INDEX][1], vision)

        dist_count = 0
        while dist_count < path[INDEX][0] and not test_obstacle_detected(list(node["prox.horizontal"])):
            await move_to(client, node, 1)
            dist_count += 1

        INDEX += 1
    return

async def angle_correction(client, node, target_angle, vision):
    rot_speed = 4.5 / np.pi

    # if target_angle > 0:
    #     await node.set_variables(motors(-100, 100))
    #     await client.sleep(target_angle * rot_speed)
    # elif target_angle < 0:
    #     await node.set_variables(motors(100, -100))
    #     await client.sleep(-target_angle * rot_speed)

    epsilon = 1
    _, _, robot_angle = get_thymio_pos(vision.get_image(vision._Vision__cap, False))
    error_angle = (target_angle - robot_angle) % (2 * np.pi)

    while error_angle > epsilon:
        print(f"Angle correction : {error_angle}")
        left_speed = epsilon * GAIN_ANGLE 
        right_speed = epsilon * GAIN_ANGLE
        await node.set_variables(motors(left_speed, right_speed))
        _, _, robot_angle = get_thymio_pos(vision.get_image(vision._Vision__cap, False))
        error_angle = (target_angle - robot_angle) % (2 * np.pi)
    await node.set_variables(motors(0, 0))

async def move_to(client, node, dist):
    print("Moving 1 unit forward")
    speed = 1 / 3.5

    await node.set_variables(motors(100, 100))
    await client.sleep(dist * speed)
    #await node.set_variables(motors(0, 0))

    return

async def motion_control(client, node, vision):
    update_state(node)
    if state == "KIDNAPPED":
        await node.set_variables(motors(0, 0))
        return True
        print("Kidnapped")
    elif state == "OBSTACLE":
        await avoid_obstacle(client, node)
        return True
    elif state == "MOVE":
        await follow_instruction(client, node, vision)
        return False
    else:
        raise("State error")