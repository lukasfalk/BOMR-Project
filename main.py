import numpy as np
import matplotlib.pyplot as plt 
import time
import cv2

import asyncio
import numpy as np
import math #pour test distance_directe
from typing import Iterable, Tuple #same as above
from tdmclient import ClientAsync

from vision import Vision
from global_nav import GlobalNavigation #from global_nav import GlobalNavigation
from filtering import Filtering
from motion_control import *
import time
from enum import Enum

v = Vision()

class State(Enum):
    GRID_CREATION = 0
    GLOBAL_NAVIGATION = 1
    KIDNAPPING = 2
    GOAL_REACHED = 3
    OBS_AVOIDED = 4

def calculate_path_positions(displacements, start_pos_cm):
    """
    Calculate path positions from displacement vectors
    
    Args:
        displacements: List of displacement vectors [(norm1, angle1), (norm2, angle2), ...]
                      where norm is in cm and angle is in radians
        start_pos_cm: Starting position (x, y) in cm
    
    Returns:
        path_positions: List of positions in cm [(x1, y1), (x2, y2), ...]
    """
    # Calculate accumulated position from displacements (norm and angle) in cm
    current_pos = np.array(start_pos_cm, dtype=float)
    positions = [current_pos.copy()]
    
    for norm, angle in displacements:
        # Convert polar coordinates (norm, angle) to cartesian (dx, dy) in cm
        dx = norm * np.cos(angle)
        dy = norm * np.sin(angle)
        current_pos += np.array([dx, dy])
        positions.append(current_pos.copy())
    
    return positions

import cv2
import numpy as np

def cm_to_px(pos_cm, cm_to_pixel, image_height):
    """Convert a position in cm to pixel coordinates."""
    x_px = int(pos_cm[0] * cm_to_pixel)
    y_px = int(image_height - pos_cm[1] * cm_to_pixel)
    return (x_px, y_px)


def visualize_realtime(
    image,
    global_path_displacements,
    start_pos_cm,
    cm_to_pixel,
    pos_measured_cm,
    pos_estimated_cm=(10, 10),
    angle_estimated_rad=None,
    angle_vision_rad=None
):
    """
    Display real-time view of camera with global path and two positions.
    Displays both estimated angle and vision angle.
    """

    image_overlay = image.copy()

    # --- Compute path positions (same as before) ---
    path_positions = calculate_path_positions(global_path_displacements, start_pos_cm)

    # Convert path positions to pixels
    path_positions_px = [
        cm_to_px(pos, cm_to_pixel, image.shape[0]) for pos in path_positions
    ]

    # --- Draw global path ---
    for i in range(len(path_positions_px) - 1):
        x1, y1 = path_positions_px[i]
        x2, y2 = path_positions_px[i+1]

        if (0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
            0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]):
            cv2.line(image_overlay, (x1, y1), (x2, y2), (0,165,255), 2)

    # --- Draw path waypoints ---
    for i, (x, y) in enumerate(path_positions_px):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            if i == 0:
                cv2.circle(image_overlay, (x, y), 6, (0,255,0), -1)       # Start
            elif i == len(path_positions_px) - 1:
                cv2.circle(image_overlay, (x, y), 6, (255,0,0), -1)      # Goal
            else:
                cv2.circle(image_overlay, (x, y), 3, (0,165,255), -1)    # Intermediate

    # --- Draw measured pos ---
    angle_offset = 0  # Adjust for image coordinate system
    if pos_measured_cm[0] is not None:
        angle_vision_rad = -angle_vision_rad
        pos_meas_px = cm_to_px(pos_measured_cm, cm_to_pixel, image.shape[0])

        if (0 <= pos_meas_px[0] < image.shape[1] and 
            0 <= pos_meas_px[1] < image.shape[0]):

            cv2.circle(image_overlay, pos_meas_px, 8, (0,0,255), 2)
            cv2.drawMarker(image_overlay, pos_meas_px, (0,0,255),
                           cv2.MARKER_CROSS, 12, 2)
            
            # ===== Draw vision angle arrow =====
            if angle_vision_rad is not None:
                arrow_length_cm = 5  # length in cm
                arrow_length_px = int(arrow_length_cm * cm_to_pixel)

                x_end = int(pos_meas_px[0] + arrow_length_px * np.cos(angle_vision_rad+angle_offset))
                y_end = int(pos_meas_px[1] - arrow_length_px * np.sin(angle_vision_rad+angle_offset))

                cv2.arrowedLine(
                    image_overlay,
                    pos_meas_px,
                    (x_end, y_end),
                    (0,0,255),  # red arrow for vision angle
                    2,
                    tipLength=0.3
                )

    # --- Draw estimated pos ---
    if pos_estimated_cm[0] is not None:
        angle_estimated_rad = -angle_estimated_rad
        pos_est_px = cm_to_px(pos_estimated_cm, cm_to_pixel, image.shape[0])

        if (0 <= pos_est_px[0] < image.shape[1] and 
            0 <= pos_est_px[1] < image.shape[0]):

            cv2.circle(image_overlay, pos_est_px, 8, (255,255,0), 2)
            cv2.drawMarker(image_overlay, pos_est_px, (255,255,0),
                           cv2.MARKER_CROSS, 12, 2)

            # ===== Draw heading arrow =====
            if angle_estimated_rad is not None:
                arrow_length_cm = 5  # length in cm
                arrow_length_px = int(arrow_length_cm * cm_to_pixel)

                x_end = int(pos_est_px[0] + arrow_length_px * np.cos(angle_estimated_rad+angle_offset))
                y_end = int(pos_est_px[1] - arrow_length_px * np.sin(angle_estimated_rad+angle_offset))

                cv2.arrowedLine(
                    image_overlay,
                    pos_est_px,
                    (x_end, y_end),
                    (255,255,0),  # yellow arrow for estimated angle
                    2,
                    tipLength=0.3
                )

    # --- Legend (unchanged) ---
    legend_y = 30
    cv2.putText(image_overlay, "Legend:", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.circle(image_overlay, (20, legend_y + 25), 5, (0,255,0), -1)
    cv2.putText(image_overlay, "Start", (35, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.circle(image_overlay, (20, legend_y + 50), 5, (255,0,0), -1)
    cv2.putText(image_overlay, "Goal", (35, legend_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.circle(image_overlay, (20, legend_y + 75), 5, (0,165,255), -1)
    cv2.putText(image_overlay, "Path", (35, legend_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.drawMarker(image_overlay, (20, legend_y + 100), (0,0,255),
                   cv2.MARKER_CROSS, 8, 2)
    cv2.putText(image_overlay, "Measured", (35, legend_y + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.drawMarker(image_overlay, (20, legend_y + 125), (255,255,0),
                   cv2.MARKER_CROSS, 8, 2)
    cv2.putText(image_overlay, "Estimated", (35, legend_y + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return image_overlay

async def main():
    #global v, ekf, pos_est, P_est, pos_pred
    GOAL_EPS_CM = 5.0 #threshold (cm) to consider the final goal reached
    WAYPOINT_EPS_CM = 6.0 #threshold (cm) to consider a waypoint reached
    SKIP_STEPS =  10 #steps to skip after local avoidance 

    try:        
        #Initialize variables for path visualization
        current_path = None  #Vector of displacement vectors at each step
        current_image = None
    
        try:
            mc = await Motion_control.create()
        except:
            print("Could not connect to Thymio")
            return
        gnav = GlobalNavigation()
        step_count = 0
        current_path = None  #Will be filled with displacement vectors
        
        just_changed_state = False  
        state = State.GRID_CREATION

        vector_path_inversed = []

        v.cam_centering()

        while(1):
            if state == State.KIDNAPPING:
                if just_changed_state:
                    cv2.destroyAllWindows()#close all windows to "restart" properly
                    await mc.client.sleep(0.1)
                    #empty the camera buffer
                    for idx in range(50):
                        _ = v.get_image(False)
                    just_changed_state = False
                    print("Kidnapped during path following")
                x,_,_ = v.get_thymio_pos(v.get_image(False))
                if x != None:
                    print("Robot detected after kidnapping")
                    await mc.client.sleep(3) #wait 3 seconds for not having the hands of the user (who did the kidnapping) in the vision/wait to stabilize

                    #empty the camera buffer
                    for idx in range(50):
                        _ = v.get_image(False) 

                    state = State.GRID_CREATION
                    just_changed_state = True

            if state == State.GRID_CREATION:
                print("Grid Creation")
                v.vision(5,50,True,10)  #acquisition delay, white threshold, plot, P (pixels per cell)
                v.plot_grid()

                #Plot the grid on a frame
                v.overlay_grid_on_cropped()
                 
                gnav.set_gnav(v)
                current_path, explored = gnav.grid_search()

                if (current_path == None) : 
                    print("No path found, obstacles probably to close")
                    break

                _ = v.get_image(False)#empty the camera buffer
                frame = v.get_cutted_frame(False, False)
                pos_est_tuple = v.get_thymio_pos_in_cm(frame)
                
                #Ensure pos_est is valid, otherwise wait until robot is detected
                while pos_est_tuple[0] is None or pos_est_tuple[1] is None or pos_est_tuple[2] is None:
                    print("Waiting for robot detection to initialize position...")
                    await mc.client.sleep(0.5)
                    _ = v.get_image(False)
                    frame = v.get_cutted_frame(False, False)
                    pos_est_tuple = v.get_thymio_pos_in_cm(frame)
                
                #Convert tuple to numpy array for the Kalman filter
                pos_est = np.array(pos_est_tuple)

                gnav.display_grid_with_path(current_path)
                gnav.display_colored_grid()

                #gnav -> find the array of vectors (deplacement at step k)
                #Example: current_path = [(norm1, theta1), (norm2, theta2), ...] representing each displacement
                #Caution: theta is in radians AND relative to the origin and not between steps -> theta_k != theta_k+1 - theta_k
                vector_path = gnav.vectors_for_displacement(current_path)

                step_count = 1

                vector_path_inversed = [] #in case of kidnapping we do not want the paths to adds up
                for idx in range(len(vector_path)):
                    norm, angle = vector_path[idx]
                    vector_path_inversed.append((norm, -angle))  #angles not in correct reference -> invert them

                current_image = v.get_cutted_frame(False,False)
                
                if current_image is not None:
                    #Get start position (cm), scale (pixels/cm) and start orientation (radians)
                    x_cm, y_cm, cm_to_pixel, _ = v.get_start_pos_and_cm_to_pixel(current_image)  

                    if cm_to_pixel is None:
                        print("Scale unavailable: skipping path plotting")
                    else:
                        if x_cm is None or y_cm is None:
                            start_pos = np.zeros(2)
                        else:
                            start_pos = np.array([x_cm, y_cm])
            
                #save scale (pixels per cm) for later conversions
                cm_to_pixel_global = cm_to_pixel

                next_step = None
                pos_to_goal = [(float(start_pos[0]), float(start_pos[1]))]
                for norm, angle in vector_path_inversed:
                    prev_x, prev_y = pos_to_goal[-1]
                    x = prev_x + norm * np.cos(angle) 
                    y = prev_y + norm * np.sin(angle)  
                    pos_to_goal.append((x, y))
                state = State.GLOBAL_NAVIGATION
                just_changed_state = True
                print(f"Pos to goal = {pos_to_goal}")

                target_pos = None
                error_pos = 0

                _ = v.get_image(False)#empty the camera buffer
                frame = v.get_cutted_frame(False, False)
                mc.set_ekf_initial_state(v.get_thymio_pos_in_cm(frame))
                mc.was_still=False
                mc.update_filtering(v)
                mc.was_still=True
                
            if state == State.GLOBAL_NAVIGATION or state == State.OBS_AVOIDED:
                _ = v.get_image(False)#empty the camera buffer
                frame = v.get_cutted_frame(False, False)
                pos_robot_vision = v.get_thymio_pos_in_cm(frame)
                angle_robot_vision = pos_robot_vision[2]

                prox_gnd = list(mc.node["prox.ground.delta"])
                if pos_robot_vision[0] is None and max(prox_gnd) < 40:#if kidnapped and it hides the marker
                    print("Kidnapped during path following")
                    state = State.KIDNAPPING
                    await mc.node.set_variables(mc.motors(0, 0,v))#stop the motors
                    just_changed_state = True
                    continue

                #update the estimated position (the one to print) with the EKF which is inside motion_control
                pos_robot_est = mc.pos_est[0], mc.pos_est[1]
                angle_robot_est =  mc.pos_est[2] 
                
                realtime_image = visualize_realtime(frame, vector_path_inversed, start_pos, 
                                                    cm_to_pixel_global, pos_robot_vision[:2], 
                                                    pos_robot_est, angle_robot_est, angle_robot_vision)
                cv2.imshow("Real-time Navigation View", realtime_image)
                if pos_robot_vision[0] is None:
                    cv2.waitKey(19)  #When the vision is lost -> slow down the visualization to not flood the camera (30 fps but motion control already slow it down)
                cv2.waitKey(1)#necessary to update the imshow window 
                

                if abs(np.subtract(pos_robot_est, pos_to_goal[-1])[0]) < GOAL_EPS_CM and abs(np.subtract(pos_robot_est, pos_to_goal[-1])[1]) < GOAL_EPS_CM:
                    state = State.GOAL_REACHED
                    
                elif step_count < len(pos_to_goal):

                    if state == State.OBS_AVOIDED:
                        if(step_count + SKIP_STEPS >= len(pos_to_goal)):
                            step_count = len(pos_to_goal) - 1
                        else:
                            step_count += SKIP_STEPS
                        state = State.GLOBAL_NAVIGATION

                    elif target_pos is not None:
                        error_pos = np.linalg.norm(target_pos) - np.linalg.norm(pos_robot_est)

                    target_pos = pos_to_goal[step_count]                        

                    dx = target_pos[0] - pos_robot_est[0]
                    dy = target_pos[1] - pos_robot_est[1]
                    norm = np.linalg.norm([dx, dy])
                    angle = -np.atan2(dy, dx)
                    next_step = (norm, angle)

                    state = await mc.fsm(next_step, error_pos, state,v)
                    mc.was_still=True

                    #Check if waypoint is passed -> if it did -> step + 1
                    while(passed_way_point(pos_robot_est, target_pos, WAYPOINT_EPS_CM)):
                        if step_count < len(pos_to_goal) - 1:
                            step_count += 1
                            target_pos = pos_to_goal[step_count]
                        else:
                            state = State.GOAL_REACHED
                            break


                elif state != State.GOAL_REACHED: # try to reach again the goal
                    step_count -= 1 

            if state == State.GOAL_REACHED:
                print(f"Goal reached")
                await mc.node.set_variables(mc.motors(0, 0,v))
                cv2.destroyAllWindows()  # Fermer toutes les fenêtres OpenCV
                
                # Plot position comparison
                plot_position_comparison(mc)
                
                break

    finally:
        cv2.destroyAllWindows()  #Close all windows
        await mc.close()

def passed_way_point(robot_pos, way_point, threshold_cm):
    '''
    Check if the robot has passed the waypoint within a certain threshold.
    '''
    distance = np.linalg.norm(np.subtract(robot_pos, way_point))
    return distance < threshold_cm

def plot_position_comparison(mc):
    '''
    Plot comparison between measured positions (mc.plot_pos) and estimated positions (mc.plot_est_pos).
    Creates 3 subplots: x position, y position, and angle over time.
    Adds gray vertical bars when vision is lost/recovered.
    '''
    if len(mc.plot_pos) == 0 or len(mc.plot_est_pos) == 0:
        print("No position data to plot")
        return
    
    # Convert lists to numpy arrays for easier manipulation
    pos_measured = np.array(mc.plot_pos)  # Shape: (N, 3) where columns are [x, y, angle]
    pos_estimated = np.array(mc.plot_est_pos)  # Shape: (M, 3)
    pos_predicted = np.array(mc.plot_pred_pos)  # Shape: (K, 3)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('Position Comparison: Measured vs Estimated', fontsize=14)
    
    # Find transitions (vision lost/recovered) based on None values in pos_measured
    vision_transitions = []
    for i in range(1, len(pos_measured)):
        curr_is_none = pos_measured[i][0] is None
        prev_is_none = pos_measured[i-1][0] is None
        if curr_is_none and not prev_is_none:
            vision_transitions.append(i-1)  # Vision perdue à l'indice i
        if not curr_is_none and prev_is_none:
            vision_transitions.append(i)  # Vision retrouvée à l'indice i
    
    # Plot X position
    axes[0].plot(range(len(pos_measured)), pos_measured[:, 0], 'b-', label='Measured X', linewidth=1.5)
    axes[0].plot(range(len(pos_estimated)), pos_estimated[:, 0], 'r--', label='Estimated X', linewidth=1.5)
    axes[0].plot(range(len(pos_predicted)), pos_predicted[:, 0], 'g-.', label='Predicted X', linewidth=1.5)
    axes[0].set_ylabel('X Position (cm)', fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Y position
    axes[1].plot(range(len(pos_measured)), pos_measured[:, 1], 'b-', label='Measured Y', linewidth=1.5)
    axes[1].plot(range(len(pos_estimated)), pos_estimated[:, 1], 'r--', label='Estimated Y', linewidth=1.5)
    axes[1].plot(range(len(pos_predicted)), pos_predicted[:, 1], 'g-.', label='Predicted Y', linewidth=1.5)
    axes[1].set_ylabel('Y Position (cm)', fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot Angle
    axes[2].plot(range(len(pos_measured)), pos_measured[:, 2], 'b-', label='Measured Angle', linewidth=1.5)
    axes[2].plot(range(len(pos_estimated)), pos_estimated[:, 2], 'r--', label='Estimated Angle', linewidth=1.5)
    axes[2].plot(range(len(pos_predicted)), pos_predicted[:, 2], 'g-.', label='Predicted Angle', linewidth=1.5)
    axes[2].set_ylabel('Angle (rad)', fontsize=11)
    axes[2].set_xlabel('Index', fontsize=11)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add gray vertical bars at vision transitions
    for idx in vision_transitions:
        for ax in axes:
            ax.axvline(x=idx, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
        asyncio.run(main())