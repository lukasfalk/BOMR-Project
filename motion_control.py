import asyncio
import numpy as np
from tdmclient import ClientAsync

from local_avoidance import *

FORWARD_SPEED = 100
GAIN = 2
GAIN_ANGLE = 60

KIDNAPPING_THR = 40

class Motion_control:
    def __init__(self):
        self.client = None
        self.node = None
        self.path_idx = 0
        self.state = "UNKOWN"

    @classmethod
    async def create(cls):
        self = cls()
        self.client = ClientAsync()
        self.node = await self.client.wait_for_node()
        await self.node.lock()
        await self.node.wait_for_variables({"prox.horizontal"})
        await self.node.wait_for_variables({"prox.ground.delta"})

        # with open("onboard_local_avoidance.txt", "r") as f:
        #     program_text = f.read()
        # await self.node.compile(program_text)
        # await self.node.run()
        return self

    async def close(self):
        try:
            if self.node is not None:
                await self.node.set_variables(self.motors(0, 0))
                await self.node.unlock()
        except Exception:
            pass

    def motors(self, l, r):
        return {"motor.left.target": [int(l)], "motor.right.target": [int(r)]}

    def update_state(self):
        prox_h = list(self.node["prox.horizontal"])
        prox_gnd = list(self.node["prox.ground.delta"])

        if self.test_kidnapping(prox_gnd):
            self.state = "KIDNAPPED"
            return
        elif test_no_obstacle(prox_h) and self.state != "MOVE":
            self.state = "MOVE"
        elif test_obstacle_detected(prox_h) and self.state != "OBSTACLE":
            self.state = "OBSTACLE"
        return

    def test_kidnapping(self, prox):
        global KIDNAPPING_THR
        return max(prox) < KIDNAPPING_THR

    async def follow_instruction(self, path, v):
        await self.path_following(path, v)

    async def path_following(self, path, v):
        #path = [(20, 0), (10, np.pi/2), (10, np.pi/2), (14.14, np.pi/4), (0, 3*np.pi/4)]
        #path = [(10, 0), (10, np.pi/2), (10, np.pi), (10, -np.pi/2)]

        while not test_obstacle_detected(list(self.node["prox.horizontal"])):
            if await self.angle_correction(path[1], v):
                return
            dist_count = 0
            while dist_count < path[0] and not test_obstacle_detected(list(self.node["prox.horizontal"])):
                await self.move_to(1)
                print(f"move dist = {dist_count}")
                dist_count += 1
            await self.node.set_variables(self.motors(0, 0))
            return

    async def angle_correction(self, target_angle, v):
        global GAIN_ANGLE
        epsilon = 0.1
        robot_detected, error_angle = self.compute_error_angle(target_angle, v)
        if not robot_detected:
            return True

        while abs(error_angle) > epsilon:
            left_speed = error_angle * GAIN_ANGLE 
            right_speed = - error_angle * GAIN_ANGLE
            await self.node.set_variables(self.motors(left_speed, right_speed))
            robot_detected, error_angle = self.compute_error_angle(target_angle, v)
            if not robot_detected:
                return True
            await self.client.sleep(0.2)
        await self.node.set_variables(self.motors(0, 0))    # 
        print("Robot aligned to target!")                   # can be deleted
        await self.client.sleep(1)                          # 
        return False
    
    def compute_error_angle(self, target, v):
        _, _, robot = v.get_thymio_pos(v.get_image(v._Vision__cap, False))
        if robot == None:
            return False, 0
        error = target - robot
        if error > np.pi:
            error = error - 2 * np.pi
        if error < -np.pi:
            error = error + 2 * np.pi
        print(f"angle target = {target}; angle robot = {robot}; error = {error}")
        return True, error

    async def move_to(self, dist):
        speed = 1 / 3.5
        await self.node.set_variables(self.motors(100, 100))
        await self.client.sleep(dist * speed)
        return

    async def fsm(self, path, v):
        self.update_state()
        if self.state == "KIDNAPPED":
            await self.node.set_variables(self.motors(0, 0))
            return True
            print("Kidnapped")
        elif self.state == "OBSTACLE":
            await avoid_obstacle(self)
            return True
        elif self.state == "MOVE":
            await self.follow_instruction(path, v)
            return False
        else:
            raise("State error")