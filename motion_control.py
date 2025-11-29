import asyncio
import numpy as np
from tdmclient import ClientAsync

from local_avoidance import *
#from vision import robot_get_thymio_pos

FORWARD_SPEED = 100
GAIN = 2
GAIN_AVD = 5
GAIN_ANGLE = 5

KIDNAPPING_THR = 40

class Motor_control:
    def __init__(self):
        self.client = None
        self.node = None
        self.path__idx = 0
        self.state = "UNKOWN"

    @classmethod
    async def create(cls):
        self = cls()
        self.client = ClientAsync()
        self.node = await self.client.wait_for_node()
        await self.node.lock()
        await self.node.wait_for_variables({"prox.horizontal"})
        await self.node.wait_for_variables({"prox.ground.delta"})
        return self

    async def close(self):
        try:
            if self.node is not None:
                await self.node.set_variables(motors(0, 0))
                await self.node.unlock()
        except Exception:
            pass

    def motors(self, l, r):
        return {"motor.left.target": [int(l)], "motor.right.target": [int(r)]}

    def test_kidnapping(self, prox):
        return max(prox) < KIDNAPPING_THR

    def update_state(self):
        prox_h = list(self.node["prox.horizontal"])
        prox_gnd = list(self.node["prox.ground.delta"])
        # print(f'Prox = {prox_h}')
        # print(f'Prox gnd = {prox_gnd}')


        if self.test_kidnapping(prox_gnd):
            self.state = "KIDNAPPED"
            #print(f'State updated to {self.state}.')
            return
        elif test_no_obstacle(prox_h) and self.state != "MOVE":
            self.state = "MOVE"
            #print(f'State updated to {self.state}.')
        elif test_obstacle_detected(prox_h) and self.state != "OBSTACLE":
            self.state = "OBSTACLE"
            #print(f'State updated to {self.state}.')
        # else:
        #     print("No state change.")

    async def follow_instruction(self):
        await self.gradient_following() # change for the instruction to follow
        #await path_following(self)

    async def gradient_following(self): # instruction for testing
        prox_gnd = list(self.node["prox.ground.delta"])
        grad = prox_gnd[1] - prox_gnd[0]
        print(f'prox = {prox_gnd} ; grad = {grad}')
        left_speed = FORWARD_SPEED - grad * GAIN
        right_speed = FORWARD_SPEED + grad * GAIN
        await self.node.set_variables(motors(left_speed, right_speed))
        await self.client.sleep(0.25)

    async def path_following(self):
        path = [(20, 0), (10, np.pi/2), (10, np.pi/2), (14.14, np.pi/4), (0, 3*np.pi/4)]
        while self.path__idx < len(path) and not test_obstacle_detected(list(self.node["prox.horizontal"])):
            await self.angle_correction(self, path[self.path__idx][1])

            dist_count = 0
            while dist_count < path[self.path__idx][0] and not test_obstacle_detected(list(self.node["prox.horizontal"])):
                await self.move_to(self, 1)
                dist_count += 1 

            self.path__idx += 1

    async def angle_correction(self, target_angle):
        rot_speed = 4.5 / np.pi

        if target_angle > 0:
            await self.node.set_variables(motors(-100, 100))
            await self.client.sleep(target_angle * rot_speed)
        elif target_angle < 0:
            await self.node.set_variables(motors(100, -100))
            await self.client.sleep(-target_angle * rot_speed)

        # epsilon = 1
        # _, _, robot_angle = robot_get_thymio_pos(Vision, v.get_image(v._Vision__cap, False))
        # error_angle = (target_angle - robot_angle) % (2 * np.pi)

        # while error_angle > epsilon:
        #     left_speed = epsilon * GAIN_ANGLE 
        #     right_speed = epsilon * GAIN_ANGLE
        #     await node.set_variables(motors(left_speed, right_speed))
        #     _, _, robot_angle = robot_get_thymio_pos(Vision, v.get_image(v._Vision__cap, False))
        #     error_angle = (target_angle - robot_angle) % (2 * np.pi)
        await self.node.set_variables(motors(0, 0))

    async def move_to(self, dist):
        speed = 1 / 3.5

        await self.node.set_variables(motors(100, 100))
        await self.client.sleep(dist * speed)
        #await node.set_variables(motors(0, 0))
    
    async def fsm(self):
        self.update_state()
        if self.state == "KIDNAPPED":
            await self.node.set_variables(motors(0, 0))
            print("Kidnapped")
        elif self.state == "OBSTACLE":
            await avoid_obstacle(self)
        elif self.state == "MOVE":
            await self.follow_instruction()
        else:
            raise("State error")

        await self.client.sleep(0.1)

async def main():
    mc = await Motor_control.create()
    try:
        while True:
            await mc.fsm()
    finally:
        await mc.close() 

if __name__ == "__main__":
    asyncio.run(main())