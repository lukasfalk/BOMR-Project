import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from typing import Tuple


class Filtering : 


    def __init__(self): 

        # Robot parameters
        self.Ts = 0.01  # Sample time
        self.interwheel_distance = 0.11 # [m] Distance between the wheels
        # Q = np.diag([10, 10, 10])  # Process noise covariance
        
        # Noise
        self.var_v_left = 2.853437746116455
        self.var_v_right = 5.5411623536575645
        self.Q = np.diag([self.var_v_left**2, self.var_v_right**2, 10]) # Process noise covariance
        self.R = np.diag([0.05**2, 0.05**2, (np.deg2rad(5))**2]) # Vision measurement noise covariance

        self.x_est = np.zeros(3) #Initial estimation for states
        self.P_est = np.diag([1e-3, 1e-3, 1e-3])  # Initial estimation covariance

        self.x = None
        self.omega = None 
        self.theta = None 



    def wrap_angle(theta: float) -> float:
        return (theta + np.pi) % (2*np.pi) - np.pi

    def state_transition_jacobian(self):
        ''' Compute the derived Jacobian of the state space model'''
        return np.array([[1, 0, -self.v * np.sin(self.theta) * self.Ts],
                         [0, 1,  self.v * np.cos(self.theta) * self.Ts],
                         [0, 0, 1]])

    def predict_state_est(self,
                        left_speed: float, right_speed: float) -> Tuple[np.ndarray, np.ndarray]:
        ''' Predict the next state and covariance'''
        self.v     = (left_speed  + right_speed) / 2 # Average speed
        self.omega = (right_speed - left_speed) / self.interwheel_distance  # Angular velocity

        #self.x[2] is theta the angle

        if abs(self.omega) < 1e-9:
            x_pred = np.array([
                        self.x[0] + self.v * np.cos(self.x[2]) * self.Ts,
                        self.x[1] + self.v * np.sin(self.x[2]) * self.Ts,
                        self.wrap_angle(self.x[2] + self.omega * self.Ts)])
        else:
            x_pred = np.array([
                        self.x[0] + (self.v / self.omega) * (np.sin(self.x[2] + self.omega * self.Ts) - np.sin(self.x[2])),
                        self.x[1] + (self.v / self.omega) * (-np.cos(self.x[2] + self.omega * self.Ts) + np.cos(self.x[2])),
                        self.wrap_angle(self.x[2] + self.omega * self.Ts)])

        F = state_transition_jacobian()

        # Process noise
        # V = np.array([
        #     [0.5 * np.cos(x[2]) * Ts, 0.5 * np.cos(x[2]) * Ts],
        #     [0.5 * np.sin(x[2]) * Ts, 0.5 * np.sin(x[2]) * Ts],
        #     [-Ts / interwheel_distance, Ts / interwheel_distance]
        # ])

        # Q = V @ R @ V.T

        P_pred = F @ P @ F.T + Q

        return x_pred, P_pred

    def update_state_est(x_pred: np.ndarray, P_pred: np.ndarray,
                        z: np.ndarray, R: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        ''' Update the state estimate with measurement z'''
        h = np.eye(3)  # Measurement model
        H = h # I know it's the same, but to conform to EKF notation
        inno = z - H @ x_pred     # Innovation
        S = H @ P_pred @ H.T + R  # Innovation covariance
        K = P_pred @ H.T @ inv(S) # Kalman gain
        x_est = x_pred + K @ inno
        P_est = (np.eye(3) - K @ H) @ P_pred
        return x_est, P_est

    def extended_kalman_filter(self,
                            left_speed: float, right_speed: float,
                            z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
            Perform one iteration of the Extended Kalman Filter.
            Takes in the previous state estimate and covariance,
            the current left and right wheel speeds,
            and returns the updated state estimate and covariance.
        '''
        x_prev = self.x_est.copy()
        P_prev = self.P_est.copy()
        # Prediction step
        x_pred, P_pred = predict_state_est(x_prev, P_prev,
                                        left_speed, right_speed,
                                        Ts, interwheel_distance, Q)
        # Update step
        if z is not None:
            x_est, P_est = update_state_est(x_pred, P_pred, z, R)
        else:
            # No measurement update. Estimated states are the predicted states
            x_est = x_pred
            P_est = P_pred
        return x_est, P_est, x_pred

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
