import asyncio
import numpy as np
from tdmclient import ClientAsync

from local_avoidance import *

FORWARD_SPEED = 100
GAIN = 2
GAIN_AVD = 5
GAIN_ANGLE = 5

KIDNAPPING_THR = 40

STATE = "UNKOWN"

INDEX = 0

robot_angle = 0 # temp, has to be replace with the vision data

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
    return max(prox) < KIDNAPPING_THR

async def follow_instruction(client, node, path_idx):
    #await gradient_following(client, node) # change for the instruction to follow
    await path_following(client, node, path_idx)
    return

async def gradient_following(client, node): # instruction for testing
    prox_gnd = list(node["prox.ground.delta"])
    grad = prox_gnd[1] - prox_gnd[0] * 0.9
    print(f'prox = {prox_gnd} ; grad = {grad}')
    left_speed = FORWARD_SPEED - grad * GAIN
    right_speed = FORWARD_SPEED + grad * GAIN
    await node.set_variables(motors(left_speed, right_speed))
    await client.sleep(0.25)
    return

async def path_following(client, node, i):
    global INDEX
    path = [(20, 0), (10, np.pi/2), (10, np.pi/2), (14.14, np.pi/4), (0, 3*np.pi/4)]
    if INDEX >= len(path) and test_obstacle_detected(list(node["prox.horizontal"])):
        return
    await move_to(client, node, path[INDEX][0], path[INDEX][1])
    INDEX += 1
    return

async def move_to(client, node, dist, angle):
    speed = 1 / 3.5
    rot_speed = 4.5 / np.pi

    epsilon = 1
    error_angle = (angle - robot_angle) % (2 * np.pi)

    # while error_angle > epsilon:
    #     left_speed = epsilon * GAIN_ANGLE 
    #     right_speed = epsilon * GAIN_ANGLE
    #     await node.set_variables(motors(left_speed, right_speed))

    if angle > 0:
        await node.set_variables(motors(-100, 100))
        await client.sleep(angle * rot_speed)
    elif angle < 0:
        await node.set_variables(motors(100, -100))
        await client.sleep(-angle * rot_speed)

    await node.set_variables(motors(100, 100))
    await client.sleep(dist * speed)
    await node.set_variables(motors(0, 0))

    return

async def main():
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    try:
        await node.wait_for_variables({"prox.horizontal"})
        await node.wait_for_variables({"prox.ground.delta"})

        path_index = 0
        while True:
            update_state(node)
            if STATE == "KIDNAPPED":
                await node.set_variables(motors(0, 0))
                print("Kidnapped")
            elif STATE == "OBSTACLE":
                await avoid_obstacle(client, node)
            elif STATE == "MOVE":
                await follow_instruction(client, node, path_index)
            else:
                raise("State error")

            await client.sleep(0.1)
    finally:
        await node.set_variables(motors(0, 0))
        await node.unlock()

if __name__ == "__main__":
    asyncio.run(main())