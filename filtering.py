import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from typing import Tuple


class Filtering : 

    def __init__(self): 

        # Robot parameters
        self.Ts = 0.1  # Sample time
        self.interwheel_distance = 11 # [cm] Distance between the wheels
        # Q = np.diag([10, 10, 10])  # Process noise covariance
        
        self.thymio_speed_to_ms = 0.3146153846153846 / 1 # m/s conversion factor

        # Noise
        self.var_v_left = 0.1#0.4 #3.8 #2.853437746116455
        self.var_v_right = 0.1#0.4 #3.8 #5.5411623536575645
        self.Q = np.diag([self.var_v_left**2, self.var_v_right**2, 0.01]) # Process noise covariance
        self.R = np.diag([0.00001**2, 0.00001**2, (np.deg2rad(5))**2]) # Vision measurement noise covariance

        self.x_est = np.zeros(3) #Initial estimation for states
        self.x_pred = None 
        self.x_prev = None

        self.P_est = np.diag([1e-5, 1e-5, 1e-5])  # Initial estimation covariance
        self.P_pred = None
        self.P_prev = None
        self.omega = None 

    def set_Ts(self, Ts: float):
        self.Ts = Ts

    def wrap_angle(self, theta: float) -> float:
        return (theta + np.pi) % (2*np.pi) - np.pi

    def state_transition_jacobian(self):
        ''' Compute the derived Jacobian of the state space model
        Note: angle is negated because vision uses image coordinates (Y down)
        while our reference frame has Y up (origin bottom-left)'''
        return np.array([[1, 0, -self.v * np.sin(-self.x_prev[2]) * self.Ts],
                         [0, 1,  self.v * np.cos(-self.x_prev[2]) * self.Ts],
                         [0, 0, 1]])

    def predict_state_est(self):
        ''' Predict the next state and covariance'''

        #self.x_prev[2] is theta the angle

        if abs(self.omega) < 1e-9:
            # Angle negated for image coordinate system (Y down in images vs Y up in our frame)
            self.x_pred = np.array([
                        self.x_prev[0] + self.v * np.cos(-self.x_prev[2]) * self.Ts,
                        self.x_prev[1] + self.v * np.sin(-self.x_prev[2]) * self.Ts,
                        self.wrap_angle(self.x_prev[2] + self.omega * self.Ts)])
        else:
            # Angle negated for image coordinate system (Y down in images vs Y up in our frame)
            theta_neg = -self.x_prev[2]
            self.x_pred = np.array([
                        self.x_prev[0] + (self.v / self.omega) * (np.sin(theta_neg + self.omega * self.Ts) - np.sin(theta_neg)),
                        self.x_prev[1] + (self.v / self.omega) * (-np.cos(theta_neg + self.omega * self.Ts) + np.cos(theta_neg)),
                        self.wrap_angle(self.x_prev[2] + self.omega * self.Ts)])

        '''
        if abs(self.x_pred[2] - self.x_prev[2]) > (1/6)*np.pi: #
            self.x_pred[2] = self.wrap_angle(self.x_prev[2])'''
        F = self.state_transition_jacobian()

        self.P_pred = F @ self.P_prev @ F.T + self.Q

    def update_state_est(self, z: np.ndarray):
        ''' Update the state estimate with measurement z'''
        H = np.eye(3)  # Measurement model
        inno = z - H @ self.x_pred     # Innovation
        S = H @ self.P_pred @ H.T + self.R  # Innovation covariance
        K = self.P_pred @ H.T @ inv(S) # Kalman gain
        self.x_est = self.x_pred + K @ inno
        self.P_est = (np.eye(3) - K @ H) @ self.P_pred

    def extended_kalman_filter(self, given_x_est: np.ndarray, given_P_est: np.ndarray, 
                               left_speed: float, right_speed: float,
                               z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
            Perform one iteration of the Extended Kalman Filter.
            Takes in the previous state estimate and covariance,
            the current left and right wheel speeds,
            and returns the updated state estimate and covariance.
        '''

        self.x_est = given_x_est
        self.P_est = given_P_est

        #left_speed = (left_speed - 5) * self.thymio_speed_to_ms
        left_speed = (left_speed) * self.thymio_speed_to_ms
        right_speed *= self.thymio_speed_to_ms

        self.x_prev = self.x_est
        self.P_prev = self.P_est

        '''
        self.v     = (left_speed  + right_speed) / 2 # Average speed
        self.omega = (right_speed - left_speed) / self.interwheel_distance  # Angular velocity
        '''
        self.v     = (right_speed  + left_speed) / 2 # Average speed
        self.omega = (left_speed - right_speed) / self.interwheel_distance  # Angular velocity
        

        # Prediction step
        self.predict_state_est()

        # Update step
        if any(x is not None for x in z):
            self.update_state_est(z)
        else:
            # No measurement update. Estimated states are the predicted states
            print(f"Measurement update without z = {z}")
            self.x_est = self.x_pred
            self.P_est = self.P_pred

        return self.x_est, self.P_est, self.x_pred

    # pos_odo = []
    # pos_filt = []
    # vision_data = []

    # k0 = 0
    # N = len(left_speed_ms)
    # vision_data = [None]*N# np.zeros((N, 3))
    # for k in range(k0, N):
    #     left_speed  = left_speed_ms[k] # <- Measure current left speed
    #     right_speed = right_speed_ms[k] # <- Measure current right speed
    #     z = vision_data[k]

    #     x_est, P_est, x_pred = extended_kalman_filter(x_est, P_est, left_speed, right_speed, z)

    #     pos_odo.append(x_pred.copy()) # raw odometry
    #     pos_filt.append(x_est.copy()) # EKF estimate

    # # -------------------- plotting --------------------
    # poses_odom = np.array(pos_odo)
    # poses_filt  = np.array(pos_filt)
    # plt.figure(figsize=(7,7))
    # plt.plot(poses_odom[:,0], poses_odom[:,1], 'b-', linewidth=0.8, label='Odometry')
    # plt.plot(poses_filt[:,0], poses_filt[:,1], 'r--', linewidth=1.0, label='EKF')
    # plt.scatter(poses_odom[0,0], poses_odom[0,1], c='b', s=50, label='Start')
    # plt.scatter(poses_odom[-1,0], poses_odom[-1,1], c='g', s=50, label='End')
    # plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.axis('equal'); plt.grid(True); plt.legend()
    # plt.title('Odometry vs EKF (2D pose)')
    # plt.show()
