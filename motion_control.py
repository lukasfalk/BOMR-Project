import asyncio
import numpy as np
from tdmclient import ClientAsync

from local_avoidance import *
#from main import update_filtering
from filtering import Filtering

from vision import Vision

import time

FORWARD_SPEED = 100
GAIN_ANGLE = 60
GAIN_FWD = 5

STEP_DT = 0.01

KIDNAPPING_THR = 40

class Motion_control:
    def __init__(self):
        self.client = None
        self.node = None
        self.path_idx = 0
        self.state = "UNKOWN"
        self.visible = True
        self.angle = None
        self.ekf = Filtering()
        self.angle_epsilon = np.deg2rad(5)  # 5 degrees in radians
        self.pos_est = np.zeros(3)
        self.P_est = np.diag([1e-5, 1e-5, 1e-5])
        self.pos_pred = np.zeros(3)
        self.first_call_filter = True
        self.last_time_filter = None
        self.delta_t_filter = None
        self.first_call = True
        self.camera_ok = True#Passed to False if we lose the camera
        #self.v = Vision()

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
                await self.node.set_variables(self.motors(0, 0,None))
                await self.node.unlock()
        except Exception:
            pass

    def set_ekf_initial_state(self, pos):
        self.pos_est = pos

    def motors(self, l, r,v):
        self.update_filtering(v)
        return {"motor.left.target": [int(l)], "motor.right.target": [int(r)]}
        #return self.set_motors(l, r)
    
    def set_motors(self, l, r):
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
        return max(prox) < KIDNAPPING_THR and not self.visible

    async def follow_instruction(self, path, error_pos,v):
        await self.path_following(path, error_pos,v)

    async def path_following(self, path, error_pos,v):
        self.last_time_filter = None #reset time for filtering (since we stopped moving)
        target_dist_steps = int(path[0])
        target_angle = path[1]
        step_count = 0
        
        #while step_count < target_dist_steps and abs(self.compute_error_angle(target_angle, self.angle)) > self.angle_epsilon:
        while step_count < target_dist_steps:
            if test_obstacle_detected(list(self.node["prox.horizontal"])):
                await self.node.set_variables(self.motors(0, 0,v))
                return
            
            #error_angle = self.compute_error_angle(target_angle, angle)
            error_angle = self.compute_error_angle(target_angle, self.angle)
            print("--- Error angle:", error_angle)

            # if not self.visible:
            #     return

            angular_spd_corr = error_angle * GAIN_ANGLE
            linear_spd_corr = error_pos * GAIN_FWD
            base_spd_gain = 1 - abs(angular_spd_corr) / FORWARD_SPEED
            #print(f"Angular spd corr = {angular_spd_corr} ; Linear spd corr = {linear_spd_corr} ; Base spd gain = {base_spd_gain}")
            left_speed = FORWARD_SPEED * base_spd_gain + angular_spd_corr + linear_spd_corr
            right_speed = FORWARD_SPEED * base_spd_gain - angular_spd_corr + linear_spd_corr
            
            left_speed = max(min(left_speed, 500), -500)
            right_speed = max(min(right_speed, 500), -500)

            await self.node.set_variables(self.motors(left_speed, right_speed,v))
            
            #await self.client.sleep(STEP_DT)
            
            step_count += 1

    def compute_error_angle(self, target, angle):
        print(f"Current angle: {angle}")
        if angle is None:
            self.visible = False
            return 0
            
        error = target - angle
        
        if error > np.pi:
            error = error - 2 * np.pi
        if error < -np.pi:
            error = error + 2 * np.pi
        return error



    async def fsm(self, path, error_pos, s,v):
        self.visible = True
        self.update_state()
        if self.state == "MOVE":
            await self.follow_instruction(path, error_pos,v)

        self.update_state()
        if self.state == "KIDNAPPED":
            await self.node.set_variables(self.motors(0, 0,v))
            return s.KIDNAPPING
        
        elif self.state == "OBSTACLE":
            await avoid_obstacle(self)
            return s.OBS_AVOIDED
        
        else:
            return s.GLOBAL_NAVIGATION
        
    #first_call_filter = True
    def update_filtering(self, v):

        #print the difference in time between two calls
        
        current_time = time.time()
        #if self.last_time_filter is None:
            #print("First filtering call - initializing time")
        #else:
        if self.last_time_filter is not None:
            self.delta_t_filter = current_time - self.last_time_filter
            print(f"------ Time since last filtering call: {self.delta_t_filter:.3f} seconds")
            self.ekf.set_Ts(self.delta_t_filter/2)
        self.last_time_filter = current_time
        

        _ = v.get_image(False)
        frame = v.get_cutted_frame(False, False)
        pos_vision_tuple = v.get_thymio_pos_in_cm(frame)
        l = self.node["motor.left.speed"]
        r = self.node["motor.right.speed"]
        print(f"left = {l}; right = {r}")
        
        # Convert vision measurement to numpy array if valid, otherwise use None array
        if pos_vision_tuple[0] is not None and pos_vision_tuple[1] is not None and pos_vision_tuple[2] is not None:
            pos_vision = np.array(pos_vision_tuple)
        else:
            pos_vision = np.array([None, None, None])
        
        # Run the Kalman filter and UPDATE GLOBAL VARIABLES
        # if self.first_call == False:#To kick -> for testing
        #     print("In first truc")
        #     pos_vision = (None, None, None)
        self.pos_est, self.P_est, self.pos_pred = self.ekf.extended_kalman_filter(self.pos_est, self.P_est, l, r, pos_vision)
        #self.first_call = False#To kick -> for testing
        #self.pos_est = v.get_thymio_pos_in_cm(frame)#To kick -> for testing
        
        x_abs, y_abs, robot_angle = v.get_thymio_pos_in_cm(frame)
        if x_abs is not None and y_abs is not None and robot_angle is not None:
            print(f"Vision - pos: ({x_abs:.2f}, {y_abs:.2f}), angle: {robot_angle:.3f}")
        else:
            print("Vision - Thymio not detected")
        print(f"update_filtering - pos: ({self.ekf.x_est[0]:.2f}, {self.ekf.x_est[1]:.2f}), angle: {self.ekf.x_est[2]:.3f}")
        if robot_angle is not None:
            print(f"Abs angle = {robot_angle}; Estimated angle = {self.pos_est[2]}")
        else:
            print(f"Abs angle = None (not detected); Estimated angle = {self.pos_est[2]}")
        if robot_angle is not None:
            self.camera_ok = True
            self.angle = robot_angle
        else:
            self.camera_ok = False
            self.angle = self.pos_est[2]
        return self.pos_est, self.P_est, self.pos_pred


