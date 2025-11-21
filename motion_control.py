import asyncio
import numpy as np
from tdmclient import ClientAsync

FORWARD_SPEED = 100
PROX_THR_HI = 20
PROX_THR_LO = 10
LOOP_DT = 0.1
GAIN = 2
GAIN_AVD = 5

def motors(l, r):
    return {"motor.left.target": [l], "motor.right.target": [r]}

def test_obstacle_detected(prox):
    return max(prox[:5]) > PROX_THR_HI

def test_no_obstacle(prox):
    return max(prox[:5]) < PROX_THR_LO

async def follow_instruction(client, node):
    obstacle_detected = False

    while not obstacle_detected:
        await gradient_following(node)          # change for the instruction to follow

        if test_obstacle_detected(list(node["prox.horizontal"])):
            obstacle_detected = True

        await client.sleep(LOOP_DT)
    return

async def avoid_obstacle(client, node):
    no_obstacle = False

    w_l = [4,  2, -2, -1, -2, 1, 0]
    w_r = [-2, -1, -2,  2,  4, 0, 1]
    w = [w_l,
         w_r]

    x = np.zeros(7) # NN input prox + memory
    y = np.zeros(2) # NN output motor commands (left, right)

    while not no_obstacle:
        prox = list(node["prox.horizontal"])

        x[5] = y[0] // 10
        x[6] = y[1] // 10
        x = np.array(prox) // 100
        y = w @ x.T

        await node.set_variables(motors(int(y[0]), int(y[1])))

        if test_no_obstacle(prox):
            no_obstacle = True

        await client.sleep(LOOP_DT)
    return

async def gradient_following(node): # instruction for testing
    prox_gnd = list(node["prox.ground.delta"])
    grad = prox_gnd[1] - prox_gnd[0]
    left_speed = FORWARD_SPEED - grad * GAIN 
    right_speed = FORWARD_SPEED + grad * GAIN
    await node.set_variables(motors(left_speed, right_speed))

async def drift_correction(node): # no use for now
    epsilon = 0 # to get from the camera. epsilon > 0 for right deviation
    left_speed = FORWARD_SPEED - epsilon * GAIN
    right_speed = FORWARD_SPEED + epsilon * GAIN
    await node.set_variables(motors(left_speed, right_speed))

async def main():
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    try:
        await node.wait_for_variables({"prox.horizontal"})
        await node.wait_for_variables({"prox.ground.delta"})

        while True:
            await follow_instruction(client, node)

            await avoid_obstacle(client, node)
    finally:
        await node.set_variables(motors(0, 0))
        await node.unlock()

if __name__ == "__main__":
    asyncio.run(main())