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

from motion_control import *
# import motion_control
# import filtering
# import local_avoidance

v = Vision()

from enum import Enum  

class State(Enum):
    GRID_CREATION = 0
    GLOBAL_NAVIGATION = 1
    KIDNAPPING = 2
    GOAL_REACHED = 3
    OBS_AVOIDED = 4

def plot_path_on_image(image, displacements, start_pos_cm, cm_to_pixel):
    """
    Draw the displacement vectors on the raw image
    
    Args:
        image: The raw image to draw on (BGR format)
        displacements: List of displacement vectors at each step [(norm1, angle1), (norm2, angle2), ...]
                      where norm is in cm and angle is in radians
        start_pos_cm: Starting position (x, y) in cm
        cm_to_pixel: Conversion factor from cm to pixels (pixels per cm)
    """
    image_with_path = image.copy()
    
    # Calculate accumulated position from displacements (norm and angle) in cm
    current_pos = np.array(start_pos_cm, dtype=float)
    print("Starting position (cm):", current_pos[0],"  ", current_pos[1])
    positions = [current_pos.copy()]
    
    for norm, angle in displacements:
        # Convert polar coordinates (norm, angle) to cartesian (dx, dy) in cm
        dx = norm * np.cos(angle)
        dy = norm * np.sin(angle)
        current_pos += np.array([dx, dy])
        positions.append(current_pos.copy())
    
    # Convert all positions from cm to pixels
    positions_px = []
    for pos in positions:
        x_px = int(pos[0] * cm_to_pixel)
        # Inverser y car les pixels augmentent vers le bas, pas vers le haut
        y_px = int(image.shape[0] - pos[1] * cm_to_pixel)
        positions_px.append((x_px, y_px))
    
    # Draw circles and lines for each position
    for i, (x, y) in enumerate(positions_px):
        # Ensure coordinates are within image bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            if i == 0:  # Start point - green
                cv2.circle(image_with_path, (x, y), 5, (0, 255, 0), -1)
            elif i == len(positions_px) - 1:  # End point - blue
                cv2.circle(image_with_path, (x, y), 5, (255, 0, 0), -1)
            else:  # Path points - red
                cv2.circle(image_with_path, (x, y), 3, (0, 0, 255), -1)
    
    # Draw lines connecting the path points
    for i in range(len(positions_px) - 1):
        x1, y1 = positions_px[i]
        x2, y2 = positions_px[i+1]
        
        if (0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
            0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]):
            cv2.line(image_with_path, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_with_path, cv2.COLOR_BGR2RGB))
    plt.title("Path Visualization on Raw Image")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.show()
    
    return image_with_path

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
    global v
    mc = await Motion_control.create()
    # threshold (cm) to consider the final goal reached
    GOAL_EPS_CM = 2.0
    SKIP_STEPS =  3

    try:

        #gnav = global_nav.GlobalNavigation()
        #path, explored, operation_count = gnav.grid_search()
        #gnav.display_grid_with_path(path)
        #gnav.display_colored_grid()
        # if path:
        #     print("A* path length =", len(path)-1, "\n", path)
        #     print("length explored:", len(explored))
        #     print("A* visualization")
        # else:
        #     print("No path found with A*")
        
        # Initialize variables for path visualization
        current_path = None  # Vector of displacement vectors at each step
        current_image = None
        cm_per_pixel_global = None
        
        gnav = GlobalNavigation()
        step_count = 0
        current_path = None  # Will be filled with displacement vectors
        
        just_changed_state = True  
        state = State.GRID_CREATION
        vector_path_inversed = []

        v.cam_centering()
        while(1):
            if state == State.KIDNAPPING:
                if just_changed_state:
                    await mc.client.sleep(0.1)
                    #empty the camera buffer
                    for idx in range(50):
                        _ = v.get_image(v._Vision__cap, False)
                    just_changed_state = False
                    print("Kidnapped during path following")
                x,_,_ = v.get_thymio_pos(v.get_image(v._Vision__cap, False))
                robot_detected = (x != None)
                if robot_detected:
                    print("Robot detected after kidnapping")
                    await mc.client.sleep(3) #wait 3 seconds for not having the hands of the user (who did the kidnapping) in the vision/wait to stabilize

                    #empty the camera buffer
                    for idx in range(50):
                        _ = v.get_image(v._Vision__cap, False) 

                    state = State.GRID_CREATION
                    just_changed_state = True

            if state == State.GRID_CREATION:
                print("Grid Creation")
                v.vision(5,80,False,10)  #acquisition delay, white threshold, plot, P (pixels per cell)
                v.plot_grid()

                # si v est une instance de Vision et que v.vision(...) a été appelé
                v.overlay_grid_on_cropped()          # ouvre une fenêtre avec la superposition

                gnav.set_gnav(v)
                current_path, explored, opertation_count = gnav.grid_search()
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

                step_count = 0

                for idx in range(len(vector_path)):
                    norm, angle = vector_path[idx]
                    vector_path_inversed.append((norm,-1*angle))  #invert angle to have the right orientation (vision has y inverted compared to robot frame)

                if vector_path is not None:
                    #current_image = v.get_image(v._Vision__cap, False)
                    current_image = v.get_cutted_frame(False,False)
                    print("image size:", current_image.shape)
                
                if current_image is not None:
                    # Plot the image with the paths (displacement vectors)
                    # Get start position (cm), scale (pixels/cm) and start orientation (radians)
                    x_cm, y_cm, cm_per_pixel, start_orientation = v.get_start_pos_and_cm_per_pixel(current_image)
                    print(f"Start pos (cm): x={x_cm}, y={y_cm}, cm_per_pixel={cm_per_pixel}, start_orientation (rad)={start_orientation}")

                    if cm_per_pixel is None:
                        print("Scale unavailable: skipping path plotting")
                    else:
                        if x_cm is None or y_cm is None:
                            #print("Start position not detected: using (0,0) as fallback")
                            start_pos = np.zeros(2)
                        else:
                            start_pos = np.array([x_cm, y_cm])

                        # Apply orientation bias: make all angles relative to the aruco horizontal
                        if start_orientation is None:
                            start_orientation = 0.0
                        
                        start_orientation = 0.0

                        vector_path_biased = []
                        for norm, angle in vector_path:
                            biased_angle = angle - start_orientation
                            vector_path_biased.append((norm, biased_angle))

                        image_with_path = plot_path_on_image(current_image, vector_path_inversed, start_pos, cm_per_pixel)
            
                # save scale (pixels per cm) for later conversions
                cm_per_pixel_global = cm_per_pixel

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

            elif state == State.GLOBAL_NAVIGATION or state == State.OBS_AVOIDED:
                #pos_to_goal = [(30, 40), (40, 40), (50, 40), (60, 40), (70, 40), (80, 40)]
                #pos_to_goal = [(30, 40), (35, 40), (40, 40), (45, 40), (50, 40), (55, 40), (60, 40), (65, 40), (70, 40), (75, 40), (80, 40)]
                #pos_to_goal = [(40, 20), (50, 30), (60, 40), (70, 50), (80, 60)]
                #pos_to_goal = [(30, 30), (35, 30), (40, 35), (45, 35), (50, 40), (55, 40), (60, 45), (65, 45), (70, 50), (75, 50), (80, 55)]
                
                # get current image and thymio position in pixels
                frame = v.get_image(v._Vision__cap, False) # first call to empty the cache
                #frame = v.get_image(v._Vision__cap, False) # second call to get the right image
                frame = v.get_cutted_frame(False)
                pos_px = v.get_thymio_pos(frame)
                if pos_px[0] != None and pos_px[1] != None and pos_px[2] != None:              
                    x_px, y_px, robot_angle = pos_px

                    # ensure we have scale (pixels per cm); try to recover if missing
                    if cm_per_pixel_global is None:
                        _, _, cm_per_pixel_fallback, _ = v.get_start_pos_and_cm_per_pixel(frame)
                        cm_per_pixel_global = cm_per_pixel_fallback

                    # convert pixel coords to cm and to bottom-left origin
                    x_cm_robot = x_px / cm_per_pixel_global
                    y_cm_robot = (frame.shape[0] - y_px) / cm_per_pixel_global
                    robot_pos = (x_cm_robot, y_cm_robot)
                    #print(f"rob pos = {robot_pos}")

                    if abs(np.subtract(robot_pos, pos_to_goal[-1])[0]) < GOAL_EPS_CM and abs(np.subtract(robot_pos, pos_to_goal[-1])[1]) < GOAL_EPS_CM:
                        print(f"Robot at {robot_pos} and goal is {pos_to_goal[-1]}")
                        state = State.GOAL_REACHED
                        just_changed_state = True
                    
                    if step_count < len(pos_to_goal):

                        if state == State.OBS_AVOIDED:
                            step_count += SKIP_STEPS
                            state = State.GLOBAL_NAVIGATION
                            just_changed_state = True

                        elif target_pos is not None:
                            error_pos = np.linalg.norm(target_pos) - np.linalg.norm(robot_pos)
                        print(f"Error pos = {error_pos} \n")

                        target_pos = pos_to_goal[step_count]

                        # If robot closer to goal than next step => go one step further
                        #print(f"Dist rob-goal = {np.linalg.norm(np.subtract(pos_to_goal[-1], robot_pos))} ; Dist target-goal = {np.linalg.norm(np.subtract(pos_to_goal[-1], target_pos))}")
                        while np.linalg.norm(np.subtract(pos_to_goal[-1], robot_pos)) < np.linalg.norm(np.subtract(pos_to_goal[-1], target_pos)):
                            if step_count + 1 >= len(pos_to_goal):
                                print("overflow pos_to_goal")
                                break
                            else:
                                step_count += 1
                                target_pos = pos_to_goal[step_count]
                                print(f"Skip step {step_count - 1}")

                        dx = target_pos[0] - robot_pos[0]
                        dy = target_pos[1] - robot_pos[1]
                        norm = np.linalg.norm([dx, dy])
                        angle = -np.atan2(dy, dx)
                        next_step = (norm, angle)
                        step_count += 1

                        print(f"Robot at {robot_pos} and going to {target_pos}")
                        print(f"Step {step_count-1} (norm, angle): {next_step}")
                        state = await mc.fsm(next_step, v, error_pos, state)

                    elif state != State.GOAL_REACHED: # try to reach again the goal
                        step_count -= 1
                else:
                    print("Kidnapped during path following (due to no robot detection)")
                    state = State.KIDNAPPING
                    #stop the motors
                    await mc.node.set_variables(mc.motors(0, 0))
                    just_changed_state = True

            if state == State.GOAL_REACHED:
                print(f"Goal reached")
                await mc.node.set_variables(mc.motors(0, 0))
                return

                # #Wait finish signal
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # break

    finally:
        await mc.close()

if __name__ == "__main__":
    asyncio.run(main())