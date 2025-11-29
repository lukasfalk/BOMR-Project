import numpy as np

PROX_THR_HI = 20
PROX_THR_LO = 10
LOOP_DT = 0.1

def motors(l, r):
    return {"motor.left.target": [int(l)], "motor.right.target": [int(r)]}

def test_obstacle_detected(prox):
    return max(prox[:5]) > PROX_THR_HI

def test_no_obstacle(prox):
    return max(prox[:5]) < PROX_THR_LO

async def avoid_obstacle(mc):
    print("Locale avoidance")
    no_obstacle = False

    w_l = [4,  2, -2, -1, -2, 1, 0]
    w_r = [-2, -1, -2,  2,  4, 0, 1]
    w = [w_l,
         w_r]

    x = np.zeros(7) # NN input prox + memory
    y = np.zeros(2) # NN output motor commands (left, right)

    while not no_obstacle:
        prox = list(mc.node["prox.horizontal"])

        x[5] = y[0] // 10
        x[6] = y[1] // 10
        x = np.array(prox) // 100
        y = w @ x.T

        await mc.node.set_variables(motors(int(y[0]), int(y[1])))

        if test_no_obstacle(prox):
            no_obstacle = True

        await mc.client.sleep(LOOP_DT)

    await mc.node.set_variables(motors(100, 100))
    await mc.client.sleep(1)
    print("End local avoidance")

    return