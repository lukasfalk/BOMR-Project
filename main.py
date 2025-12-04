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
# import motion_control
# import filtering
# import local_avoidance
import time
from enum import Enum

v = Vision()
'''
ekf = Filtering()

pos_est = np.zeros(3)
P_est = np.diag([1e-3, 1e-3, 1e-3])
'''

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

def visualize_realtime(image, global_path_displacements, start_pos_cm, cm_to_pixel, 
                       pos_measured_cm, pos_estimated_cm=(10, 10), 
                       angle_measured=None, angle_estimated=None):
    """
    Display real-time view of camera with global path, two positions, and orientation arrows.
    
    Args:
        image: The cropped camera image (BGR format)
        global_path_displacements: List of displacement vectors [(norm1, angle1), ...]
        start_pos_cm: Starting position (x, y) in cm
        cm_to_pixel: Conversion factor from cm to pixels (pixels per cm)
        pos_measured_cm: Position measured with get_thymio_pos_cm (x, y) in cm
        pos_estimated_cm: Second position (x, y) in cm (default (10, 10))
        angle_measured: Orientation of the measured robot in radians (default None)
        angle_estimated: Orientation of the estimated robot in radians (default None)
    
    Returns:
        image_with_overlay: Image with path, positions, and angles drawn
    """
    image_overlay = image.copy()
    
    # Calculate path positions from displacements using the helper function
    path_positions = calculate_path_positions(global_path_displacements, start_pos_cm)
    
    # Convert path positions from cm to pixels
    def cm_to_px(pos_cm):
        x_px = int(pos_cm[0] * cm_to_pixel)
        y_px = int(image.shape[0] - pos_cm[1] * cm_to_pixel)
        return (x_px, y_px)
    
    path_positions_px = [cm_to_px(pos) for pos in path_positions]
    
    # Draw global path
    for i in range(len(path_positions_px) - 1):
        x1, y1 = path_positions_px[i]
        x2, y2 = path_positions_px[i+1]
        
        if (0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
            0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]):
            cv2.line(image_overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange path
    
    # Draw path waypoints
    for i, (x, y) in enumerate(path_positions_px):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            if i == 0:  # Start -> green
                cv2.circle(image_overlay, (x, y), 6, (0, 255, 0), -1)
            elif i == len(path_positions_px) - 1:  # Goal -> blue
                cv2.circle(image_overlay, (x, y), 6, (255, 0, 0), -1)
            else:  # Intermediate waypoints -> small orange circles
                cv2.circle(image_overlay, (x, y), 3, (0, 165, 255), -1)
    
    arrow_length = 25  # Length of the orientation arrow in pixels

    # Draw measured position (red circle with cross and arrow)
    if pos_measured_cm[0] is not None and pos_measured_cm[1] is not None:
        pos_measured_px = cm_to_px(pos_measured_cm)
        if (0 <= pos_measured_px[0] < image.shape[1] and 
            0 <= pos_measured_px[1] < image.shape[0]):
            
            # Draw Marker
            cv2.circle(image_overlay, pos_measured_px, 8, (0, 0, 255), 2)  # Red circle
            cv2.drawMarker(image_overlay, pos_measured_px, (0, 0, 255), 
                           cv2.MARKER_CROSS, 12, 2)  # Red cross
            
            # Draw Orientation Arrow
            if angle_measured is not None:
                # Note: y is subtracted because pixel coordinates increase downwards,
                # while standard angle is counter-clockwise from positive x (Right)
                end_x = int(pos_measured_px[0] + arrow_length * math.cos(angle_measured))
                end_y = int(pos_measured_px[1] - arrow_length * math.sin(angle_measured))
                cv2.arrowedLine(image_overlay, pos_measured_px, (end_x, end_y), 
                                (0, 0, 255), 2, tipLength=0.3)
    
    # Draw estimated position (cyan circle with cross and arrow)
    if pos_estimated_cm[0] is not None and pos_estimated_cm[1] is not None:
        pos_estimated_px = cm_to_px(pos_estimated_cm)
        if (0 <= pos_estimated_px[0] < image.shape[1] and 
            0 <= pos_estimated_px[1] < image.shape[0]):
            
            # Draw Marker
            cv2.circle(image_overlay, pos_estimated_px, 8, (255, 255, 0), 2)  # Cyan circle
            cv2.drawMarker(image_overlay, pos_estimated_px, (255, 255, 0), 
                           cv2.MARKER_CROSS, 12, 2)  # Cyan cross

            # Draw Orientation Arrow
            if angle_estimated is not None:
                end_x = int(pos_estimated_px[0] + arrow_length * math.cos(angle_estimated))
                end_y = int(pos_estimated_px[1] - arrow_length * math.sin(angle_estimated))
                cv2.arrowedLine(image_overlay, pos_estimated_px, (end_x, end_y), 
                                (255, 255, 0), 2, tipLength=0.3)
    
    # Add legend
    legend_y = 30
    cv2.putText(image_overlay, "Legend:", (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(image_overlay, (20, legend_y + 25), 5, (0, 255, 0), -1)
    cv2.putText(image_overlay, "Start", (35, legend_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(image_overlay, (20, legend_y + 50), 5, (255, 0, 0), -1)
    cv2.putText(image_overlay, "Goal", (35, legend_y + 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(image_overlay, (20, legend_y + 75), 5, (0, 165, 255), -1)
    cv2.putText(image_overlay, "Path", (35, legend_y + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.drawMarker(image_overlay, (20, legend_y + 100), (0, 0, 255), 
                   cv2.MARKER_CROSS, 8, 2)
    cv2.putText(image_overlay, "Measured", (35, legend_y + 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.drawMarker(image_overlay, (20, legend_y + 125), (255, 255, 0), 
                   cv2.MARKER_CROSS, 8, 2)
    cv2.putText(image_overlay, "Estimated", (35, legend_y + 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image_overlay

def distance_directe(path: Iterable[Tuple[float, float]], degrees: bool = False) -> float:
    """
    Calcule la distance directe entre l'origine et la position finale
    après avoir appliqué les déplacements donnés par (norm, angle).
    - path: iterable de (norm, angle)
    - degrees: True si les angles sont en degrés (sinon radians)
    """
    x = y = 0.0
    to_rad = math.radians if degrees else (lambda a: a)
    for r, theta in path:
        a = to_rad(theta)
        x += r * math.cos(a)
        y += r * math.sin(a)
    return math.hypot(x, y)

def displacement_angle_to_origin_angle(path):
    '''
    Convert a path defined by displacement vectors (norm, angle) to absolute angles from origin.
    '''
    abs_path = []
    current_angle = 0.0
    for norm, angle in path:
        current_angle += angle
        # Normalize to [-pi, pi]
        normalized = (current_angle + np.pi) % (2*np.pi) - np.pi
        abs_path.append((norm, normalized))
    return abs_path

#TODO:
# passer le tableau de vecteur de déplacement en (x,y)_k (soit garder le bordel dans le main pour l'instant soit faire une vraie fonction)
# finir le controlleur (ATTENTION -> angle pas dans le même repère -> cf. au fond de vision.py pour des fonctions de test de l'angle) (orientation caméra -> texte lisible depuis la map)
# sortie avoidance -> le faire + fct qui trouve le next step le plus proche de la position actuelle (pas besoin de partir depuis le début du path mais depuis le dernier step_count (i.e. celui avant avoidance))
# -> faire un step_count bien fait dans le main
async def main():
    global v, ekf, pos_est, P_est, pos_pred
    # threshold (cm) to consider the final goal reached
    GOAL_EPS_CM = 5.0
    SKIP_STEPS =  3 

    try:        
        # Initialize variables for path visualization
        current_path = None  # Vector of displacement vectors at each step
        current_image = None
        cm_per_pixel_global = None
    
        try:
            mc = await Motion_control.create()
        except:
            print("Could not connect to Thymio")
            return
        gnav = GlobalNavigation()
        step_count = 0
        current_path = None  # Will be filled with displacement vectors
        
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
                        _ = v.get_image(v._Vision__cap, False)
                    just_changed_state = False
                    print("Kidnapped during path following")
                x,_,_ = v.get_thymio_pos(v.get_image(v._Vision__cap, False))
                if x != None:
                    print("Robot detected after kidnapping")
                    await mc.client.sleep(3) #wait 3 seconds for not having the hands of the user (who did the kidnapping) in the vision/wait to stabilize

                    #empty the camera buffer
                    for idx in range(50):
                        _ = v.get_image(v._Vision__cap, False) 

                    state = State.GRID_CREATION
                    just_changed_state = True

            if state == State.GRID_CREATION:
                print("Grid Creation")
                v.vision(5,20,False,10)  #acquisition delay, white threshold, plot, P (pixels per cell)
                v.plot_grid()

                # si v est une instance de Vision et que v.vision(...) a été appelé
                v.overlay_grid_on_cropped()          # ouvre une fenêtre avec la superposition
                 
                gnav.set_gnav(v)
                current_path, explored = gnav.grid_search()

                if (current_path == None) : 
                    print("No path found, obstacles probably to close")
                    break

                _ = v.get_image(v._Vision__cap, False)#empty the camera buffer
                frame = v.get_cutted_frame(False, False)
                pos_est_tuple = v.get_thymio_pos_in_cm(frame)
                
                # Ensure pos_est is valid, otherwise wait until robot is detected
                while pos_est_tuple[0] is None or pos_est_tuple[1] is None or pos_est_tuple[2] is None:
                    print("Waiting for robot detection to initialize position...")
                    await mc.client.sleep(0.5)
                    _ = v.get_image(v._Vision__cap, False)
                    frame = v.get_cutted_frame(False, False)
                    pos_est_tuple = v.get_thymio_pos_in_cm(frame)
                
                # Convert tuple to numpy array for the Kalman filter
                pos_est = np.array(pos_est_tuple)
                print(f"Initial position detected: {pos_est}")

                gnav.display_grid_with_path(current_path)
                gnav.display_colored_grid()
                
                print("A* path length =", len(current_path)-1, "\n", current_path)

                # gnav -> find the array of vectors (deplacement at step k)
                # Example: current_path = [(norm1, theta1), (norm2, theta2), ...] representing each displacement
                #Caution: theta is in radians AND relative to the origin and not between steps -> theta_k != theta_k+1 - theta_k
                vector_path = gnav.vectors_for_displacement(current_path)
                print("Vector path 0 (norm, angle):", vector_path)
                distance_directe_result = distance_directe(vector_path, degrees=False)
                print(f"Direct distance to goal after following path: {distance_directe_result:.2f} cm")

                step_count = 1

                vector_path_inversed = [] #in case of kidnapping we do not want the paths to adds up
                for idx in range(len(vector_path)):
                    norm, angle = vector_path[idx]
                    vector_path_inversed.append((norm, -angle))  # angles already in correct reference frame

                if vector_path is not None:
                    #current_image = v.get_image(v._Vision__cap, False)
                    current_image = v.get_cutted_frame(False,False)
                    print("image size:", current_image.shape)
                
                if current_image is not None:
                    # Get start position (cm), scale (pixels/cm) and start orientation (radians)
                    x_cm, y_cm, cm_to_pixel, _ = v.get_start_pos_and_cm_to_pixel(current_image)  

                    if cm_to_pixel is None:
                        print("Scale unavailable: skipping path plotting")
                    else:
                        if x_cm is None or y_cm is None:
                            #print("Start position not detected: using (0,0) as fallback")
                            start_pos = np.zeros(2)
                        else:
                            start_pos = np.array([x_cm, y_cm])
            
                # save scale (pixels per cm) for later conversions
                cm_to_pixel_global = cm_to_pixel

                #vector_path_inversed = [(0, 0), (0, np.pi),(0, np.pi),(0, np.pi)]
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
                visualizing_counter = 0

                _ = v.get_image(v._Vision__cap, False)#empty the camera buffer
                frame = v.get_cutted_frame(False, False)
                mc.set_ekf_initial_state(v.get_thymio_pos_in_cm(frame))
                _, _, _ = update_filtering(mc)
                
            if state == State.GLOBAL_NAVIGATION or state == State.OBS_AVOIDED:
                #pos_to_goal = [(30, 40), (40, 40), (50, 40), (60, 40), (70, 40), (80, 40)]
                #pos_to_goal = [(30, 40), (35, 40), (40, 40), (45, 40), (50, 40), (55, 40), (60, 40), (65, 40), (70, 40), (75, 40), (80, 40)]
                #pos_to_goal = [(40, 20), (50, 30), (60, 40), (70, 50), (80, 60)]
                #pos_to_goal = [(30, 30), (35, 30), (40, 35), (45, 35), (50, 40), (55, 40), (60, 45), (65, 45), (70, 50), (75, 50), (80, 55)]

                # Update filtering BEFORE using pos_est in this iteration
                #update_filtering(mc)

                #TODO: Do not go to kidnapping state if vision is done -> use EKF estimation instead
                #TODO: go to kidnapping state only if both vision and motion control do not have a ground anymore
                # if pos_robot_vision[0] is None:#if kidnapped and it hides the   marker
                #     print("Kidnapped during path following (due to no robot detection)")
                #     state = State.KIDNAPPING
                #     await mc.node.set_variables(mc.motors(0, 0))#stop the motors
                #     just_changed_state = True
                #     continue

                _ = v.get_image(v._Vision__cap, False)#empty the camera buffer
                frame = v.get_cutted_frame(False, False)
                pos_robot_vision = v.get_thymio_pos_in_cm(frame)[:2]

                # Update the Kalman filter with current measurements
                '''
                pos_est = ekf.x_est
                P_est = ekf.P_est
                pos_pred = ekf.x_pred'''

                pos_est = mc.pos_est

                #pos_robot_est = pos_est[0], pos_est[1] 
                pos_est2 = mc.ekf.x_est#To kick -> for testing otherwise PUT pos_est from mc
                pos_robot_est = pos_est2[0], pos_est2[1] 
                angle_robot_est =  pos_est[2] 
                
                # if visualizing_counter >= 2:
                #     # Display real-time visualization
                #     realtime_image = visualize_realtime(frame, vector_path_inversed, start_pos, 
                #                                     cm_to_pixel_global, pos_robot_vision, pos_robot_est)
                #     cv2.imshow("Real-time Navigation View", realtime_image)
                #     cv2.waitKey(1)  # Afficher pendant 1ms pour permettre la mise à jour
                #     visualizing_counter = 0
                # else:
                #     visualizing_counter += 1
                realtime_image = visualize_realtime(frame, vector_path_inversed, start_pos, 
                                                    cm_to_pixel_global, pos_robot_vision, pos_robot_est)
                cv2.imshow("Real-time Navigation View", realtime_image)
                if pos_robot_vision[0] is None:
                    cv2.waitKey(20)  # Afficher pendant 20ms pour permettre la mise à jour
                else:
                    cv2.waitKey(1)  # Afficher pendant 1ms pour permettre la mise à jour
                print("Visualisation")
                

                if abs(np.subtract(pos_robot_est, pos_to_goal[-1])[0]) < GOAL_EPS_CM and abs(np.subtract(pos_robot_est, pos_to_goal[-1])[1]) < GOAL_EPS_CM:
                    print(f"Robot at {pos_robot_est} and goal is {pos_to_goal[-1]}")
                    state = State.GOAL_REACHED
                    
                elif step_count < len(pos_to_goal):

                    if state == State.OBS_AVOIDED:
                        step_count += SKIP_STEPS
                        state = State.GLOBAL_NAVIGATION

                    elif target_pos is not None:
                        error_pos = np.linalg.norm(target_pos) - np.linalg.norm(pos_robot_est)
                    #print(f"Error pos = {error_pos} \n")

                    target_pos = pos_to_goal[step_count]

                    # If robot closer to goal than next step => go one step further
                    #print(f"Dist rob-goal = {np.linalg.norm(np.subtract(pos_to_goal[-1], robot_pos))} ; Dist target-goal = {np.linalg.norm(np.subtract(pos_to_goal[-1], target_pos))}")
                    '''while np.linalg.norm(np.subtract(pos_to_goal[-1], pos_robot_est)) < np.linalg.norm(np.subtract(pos_to_goal[-1], target_pos)):
                        if step_count + 1 >= len(pos_to_goal):
                            print("overflow pos_to_goal")
                            break
                        else:
                            step_count += 1
                            target_pos = pos_to_goal[step_count]
                            print(f"Skip step {step_count - 1}")
                            '''

                    dx = target_pos[0] - pos_robot_est[0]
                    dy = target_pos[1] - pos_robot_est[1]
                    norm = np.linalg.norm([dx, dy])
                    angle = -np.atan2(dy, dx)
                    next_step = (norm, angle)

                    print(f"Robot at {pos_robot_est} and going to {target_pos}")
                    print(f"Step {step_count-1} (norm, angle): {next_step}")
                    #state = await mc.fsm(next_step, error_pos, state, angle_robot_est)
                    state = await mc.fsm(next_step, error_pos, state)

                    #Check if waypoint is passed -> if it did -> step + 1
                    if(passed_way_point(pos_robot_est, target_pos, GOAL_EPS_CM)):
                        if step_count < len(pos_to_goal):
                            step_count += 1
                        else:
                            state = State.GOAL_REACHED


                elif state != State.GOAL_REACHED: # try to reach again the goal
                    step_count -= 1 

            if state == State.GOAL_REACHED:
                print(f"Goal reached")
                await mc.node.set_variables(mc.motors(0, 0))
                cv2.destroyAllWindows()  # Fermer toutes les fenêtres OpenCV
                break

                # #Wait finish signal
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # break

    finally:
        cv2.destroyAllWindows()  # Assurez que les fenêtres sont fermées
        await mc.close()

'''
Check if the robot has passed the waypoint within a certain threshold.
'''
def passed_way_point(robot_pos, way_point, threshold_cm):
    distance = np.linalg.norm(np.subtract(robot_pos, way_point))
    return distance < threshold_cm

#first_call_filter = True
def update_filtering(mc):
    global v

    #print the difference in time between two calls
    
    current_time = time.time()
    #if mc.last_time_filter is None:
        #print("First filtering call - initializing time")
    #else:
    if mc.last_time_filter is not None:
        mc.delta_t_filter = current_time - mc.last_time_filter
        print(f"------ Time since last filtering call: {mc.delta_t_filter:.3f} seconds")
        mc.ekf.set_Ts(mc.delta_t_filter/2)
    mc.last_time_filter = current_time
    

    _ = v.get_image(v._Vision__cap, False)
    frame = v.get_cutted_frame(False, False)
    pos_vision_tuple = v.get_thymio_pos_in_cm(frame)
    l = mc.node["motor.left.speed"]
    r = mc.node["motor.right.speed"]
    print(f"left = {l}; right = {r}")
    
    # Convert vision measurement to numpy array if valid, otherwise use None array
    if pos_vision_tuple[0] is not None and pos_vision_tuple[1] is not None and pos_vision_tuple[2] is not None:
        pos_vision = np.array(pos_vision_tuple)
    else:
        pos_vision = np.array([None, None, None])
    
    # Run the Kalman filter and UPDATE GLOBAL VARIABLES
    if mc.first_call == False:#To kick -> for testing
        pos_vision = (None, None, None)
    mc.pos_est, mc.P_est, mc.pos_pred = mc.ekf.extended_kalman_filter(mc.pos_est, mc.P_est, l, r, pos_vision)
    mc.first_call = False#To kick -> for testing
    mc.pos_est = v.get_thymio_pos_in_cm(frame)#To kick -> for testing
    
    x_abs, y_abs, robot = v.get_thymio_pos_in_cm(frame)
    if x_abs is not None and y_abs is not None and robot is not None:
        print(f"Vision - pos: ({x_abs:.2f}, {y_abs:.2f}), angle: {robot:.3f}")
    else:
        print("Vision - Thymio not detected")
    print(f"update_filtering - pos: ({mc.ekf.x_est[0]:.2f}, {mc.ekf.x_est[1]:.2f}), angle: {mc.ekf.x_est[2]:.3f}")
    mc.angle = mc.pos_est[2]
    return mc.pos_est, mc.P_est, mc.pos_pred

if __name__ == "__main__":
        asyncio.run(main())
