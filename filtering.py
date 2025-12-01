
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from typing import Tuple

## Robot parameters
Ts = 0.01  # Sample time
interwheel_distance = 0.11 # [m] Distance between the wheels

# Q = np.diag([10, 10, 10])  # Process noise covariance

## Noise
var_v_left = 2.853437746116455
var_v_right = 5.5411623536575645
Q = np.diag([var_v_left**2, var_v_right**2, 10]) # Process noise covariance
R = np.diag([0.05**2, 0.05**2, (np.deg2rad(5))**2]) # Vision measurement noise covariance

x_est = np.zeros(3)
P_est = np.diag([1e-3, 1e-3, 1e-3])  # Initial estimation covariance

def wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2*np.pi) - np.pi

def state_transition_jacobian(v: float, theta: float) -> np.ndarray:
    ''' Compute the derived Jacobian of the state space model'''
    return np.array([[1, 0, -v * np.sin(theta) * Ts],
                     [0, 1,  v * np.cos(theta) * Ts],
                     [0, 0, 1]])

def predict_state_est(x: np.ndarray, P: np.ndarray,
                      left_speed: float, right_speed: float,
                      Ts: float, interwheel_distance: float, Q: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    ''' Predict the next state and covariance'''
    v     = (left_speed  + right_speed) / 2 # Average speed
    omega = (right_speed - left_speed) / interwheel_distance  # Angular velocity
    theta = x[2] # Angle

    if abs(omega) < 1e-9:
        x_pred = np.array([
                       x[0] + v * np.cos(theta) * Ts,
                       x[1] + v * np.sin(theta) * Ts,
            wrap_angle(x[2] + omega * Ts)
        ])
    else:
        x_pred = np.array([
                       x[0] + (v / omega) * (np.sin(theta + omega * Ts) - np.sin(theta)),
                       x[1] + (v / omega) * (-np.cos(theta + omega * Ts) + np.cos(theta)),
            wrap_angle(x[2] + omega * Ts)
        ])

    F = state_transition_jacobian(v, theta)

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

def extended_kalman_filter(x_est: np.ndarray, P_est: np.ndarray,
                           left_speed: float, right_speed: float,
                           z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Perform one iteration of the Extended Kalman Filter.
        Takes in the previous state estimate and covariance,
        the current left and right wheel speeds,
        and returns the updated state estimate and covariance.
    '''
    x_prev = x_est.copy()
    P_prev = P_est.copy()
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
