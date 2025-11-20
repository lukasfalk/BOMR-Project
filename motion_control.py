import asyncio
import numpy as np
from tdmclient import ClientAsync

FORWARD_SPEED = 50
PROX_THR_HI = 1500
PROX_THR_LO = 1000
LOOP_DT = 0.1
GAIN = 5
GAIN_AVD = 10

STATE = "FORWARD"

def motors(l, r):
    return {"motor.left.target": [l], "motor.right.target": [r]}

async def set_motors(node, l, r):
    await node.set_variables(motors(l, r))

def update_state(prox):
    global STATE
    if max(prox) > PROX_THR_HI and STATE == "FORWARD":
        STATE = "AVOIDANCE"
    elif max(prox) < PROX_THR_LO and STATE == "AVOIDANCE":
        STATE = "FORWARD"

def update_path(prox_gnd):
    global FORWARD_SPEED
    diff = prox_gnd[1] - prox_gnd[0]
    motor_left_target = FORWARD_SPEED - diff * GAIN 
    motor_right_target = FORWARD_SPEED + diff * GAIN
    return motor_left_target, motor_right_target

def avoid_obstacle(prox):
    global FORWARD_SPEED

    motor_speed_left = FORWARD_SPEED + sum(prox[:2]) * GAIN_AVD // 100
    motor_speed_right = FORWARD_SPEED + sum(prox[3:5]) * GAIN_AVD // 100

    return motor_speed_left, motor_speed_right

async def main():
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    try:
        await node.wait_for_variables({"prox.horizontal"})
        await node.wait_for_variables({"prox.ground.delta"})
        while True:
            prox = list(node["prox.horizontal"])
            prox_gnd = list(node["prox.ground.delta"])
            update_state(prox)
            print(f"Prox: {prox}")
            if STATE == "FORWARD":
                 motor_speed_left, motor_speed_right = update_path(prox_gnd)
            if STATE == "AVOIDANCE":
                motor_speed_left, motor_speed_right = avoid_obstacle(prox)

            await set_motors(node, motor_speed_left, motor_speed_right)
            await client.sleep(LOOP_DT)
    finally:
        await set_motors(node, 0, 0)
        await node.unlock()

if __name__ == "__main__":
    asyncio.run(main())
    

# async def control_loop(client, node):
#     # ensure we can read proximity values
#     await node.wait_for_variables({"prox.horizontal"})
#     # start forward
#     await set_motors(node, FORWARD_SPEED, FORWARD_SPEED)

#     try:
#         while True:
#             prox = np.array(list(node["prox.horizontal"]))  # cached values updated by client
#             prox_weights = np.array([1, 2, 3, 2, 1])
#             prox_weighted_values = prox[1:6] * prox_weights
#             motor_speed_right = FORWARD_SPEED
#             motor_speed_left = FORWARD_SPEED
#             print(f"Prox: {prox}")
#             if max(prox[:2]) > PROX_THR_HI:
#                 motor_speed_right -= sum(prox_weighted_values[:2]) // 25
#                 motor_speed_left += sum(prox_weighted_values[:2]) // 25
#             elif max(prox[2:]) > PROX_THR_HI:
#                 motor_speed_right += sum(prox_weighted_values[2:]) // 25
#                 motor_speed_left -= sum(prox_weighted_values[2:]) // 25
            
#             await set_motors(node, motor_speed_left, motor_speed_right)

#             await client.sleep(LOOP_DT)
#     finally:
#         # safety stop on exit
#         await set_motors(node, 0, 0)

# with ClientAsync() as client:
#     async def update_motor_speed(left_motor_speed, right_motor_speed):
#         with await client.lock() as node:
#             await node.set_variables(motors(left_motor_speed, right_motor_speed))
    
#     async def stop_motors():
#         with await client.lock() as node:
#             await node.set_variables(motors(0, 0))

#     client.run_async_program(lambda: update_motor_speed(80, 0))
#     client.sleep(2000)
#     client.run_async_program(stop_motors)

# try:
    #     prox = variables["prox.horizontal"]
    #     prox_weights = [0.5, 1.0, 1.5, 1.0, 0.5]
    #     prox_weighted_values = prox * prox_weights
    #     left_speed = 
    #     node.send_set_variables(motors(left_speed, right_speed))
    # except KeyError:
    #     pass  # prox.horizontal not found