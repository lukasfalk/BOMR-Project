import numpy as np
import matplotlib.pyplot as plt 
import time
import cv2

from vision import Vision
from global_nav import GlobalNavigation #from global_nav import GlobalNavigation
# import motion_control
# import filtering
# import local_avoidance

from enum import Enum   
class State(Enum):
    GRID_CREATION = 0
    GLOBAL_NAVIGATION = 1
    KIDNAPPING = 2
    GOAL_REACHED = 3


#scale is the number of pixels per grid cell
def plot_path_on_image_v0(image, displacements, scale):
    """
    Draw the displacement vectors on the raw image
    
    Args:
        image: The raw image to draw on (BGR format)
        displacements: List of displacement vectors at each step [(norm1, angle1), (norm2, angle2), ...]
                      where norm is the magnitude and angle is in radians
        scale: Scale factor for visualization (pixels per unit)
    """
    image_with_path = image.copy()
    
    # Calculate accumulated position from displacements (norm and angle)
    current_pos = np.array([0.0, 0.0])
    positions = [current_pos.copy()]
    
    for norm, angle in displacements:
        # Convert polar coordinates (norm, angle) to cartesian (dx, dy)
        dx = norm * np.cos(angle)
        dy = norm * np.sin(angle)
        current_pos += np.array([dx, dy])
        positions.append(current_pos.copy())
    
    # Draw circles and lines for each position
    for i, pos in enumerate(positions):
        x, y = int(pos[0] * scale), int(pos[1] * scale)
        
        # Ensure coordinates are within image bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            if i == 0:  # Start point - green
                cv2.circle(image_with_path, (x, y), 5, (0, 255, 0), -1)
            elif i == len(positions) - 1:  # End point - blue
                cv2.circle(image_with_path, (x, y), 5, (255, 0, 0), -1)
            else:  # Path points - red
                cv2.circle(image_with_path, (x, y), 3, (0, 0, 255), -1)
    
    # Draw lines connecting the path points
    for i in range(len(positions) - 1):
        x1, y1 = int(positions[i][0] * scale), int(positions[i][1] * scale)
        x2, y2 = int(positions[i+1][0] * scale), int(positions[i+1][1] * scale)
        
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

def main():
    state = State.GRID_CREATION
    #v = vision.Vision()
    #v.vision_test(20)

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
    v = Vision()
    gnav = GlobalNavigation()
    step_count = 0
    current_path = None  # Will be filled with displacement vectors
    
    just_changed_state = True
    while(1):

        if state == State.GRID_CREATION:
            print("Grid Creation")
            v.cam_centering()
            v.vision(5,90,True)
            v.plot_grid()
            gnav.set_gnav(v)
            current_path, explored, opertation_count = gnav.grid_search()
            gnav.display_grid_with_path(current_path)
            gnav.display_colored_grid()
            print("A* path length =", len(current_path)-1, "\n", current_path)
            vector_path = gnav.vectors_for_displacement(current_path)
            # TODO: gnav -> find the array of vectors (deplacement at step k)
            # current_path should be a list of displacement vectors (or steps)
            # Example: current_path = [(norm1, theta1), (norm2, theta2), ...] representing each displacement
            step_count = 0
            
            
            state = State.GLOBAL_NAVIGATION

        elif state == State.GLOBAL_NAVIGATION:

            if just_changed_state:
                print("Entering Global Navigation State")
                just_changed_state = False


            # TODO: update the image with the path to show that the robot is following it
            # This replots at each time step to show robot progress
            # or
            # TODO: plot the image with the paths
            # Get raw image and plot path at each step
            if vector_path is not None:
                current_image = v.get_image(v._Vision__cap, False)
                print("image size:", current_image.shape)
                
                if current_image is not None:
                    # Plot the image with the paths (displacement vectors)
                    # Get start position (cm), scale (pixels/cm) and start orientation (radians)
                    x_cm, y_cm, cm_per_pixel, start_orientation = v.get_start_pos_and_cm_per_pixel(current_image)

                    if cm_per_pixel is None:
                        print("Scale unavailable: skipping path plotting")
                    else:
                        if x_cm is None or y_cm is None:
                            print("Start position not detected: using (0,0) as fallback")
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

                        image_with_path = plot_path_on_image(current_image, vector_path_biased, start_pos, cm_per_pixel)
                        print(f"Step {step_count}: Following path, {len(vector_path_biased)} displacement vectors (angles biased by {start_orientation:.3f} rad)")
                
                step_count += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
            #is one if the proximity sensors doesn't detect anything under the robot
            #-> TODO: get this info from motion_control
            if floor_not_detected:
                state = State.KIDNAPPING
                just_changed_state = True
            #-> TODO: get this info from motion_control/if we are doing the steps computation here -> do it here
            if reached:
                state = State.GOAL_REACHED
        elif state == State.KIDNAPPING:
            if just_changed_state:
                just_changed_state = False
                print("Entering Kidnapping State")
                # TODO: kill the image with the path to show that the robot is lost
                current_path = None
                step_count = 0

            robot_detected = v.get_thymio_pos() is not None

            #-> TODO: get this info from motion_control
            if robot_detected and floor_detected:
                time.sleep(3) #wait 3 seconds for not having the hands of the user (who did the kidnapping) in the vision/wait to stabilize
                state = State.GRID_CREATION
                just_changed_state = True
        elif state == State.GOAL_REACHED:
            print("Goal Reached State")
            #potentialy do some celebration and show on the screen the mistake due to uncertainties (or just say it at the presentation)
            # TODO: STOP robot

            #Wait finish signal
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    main()