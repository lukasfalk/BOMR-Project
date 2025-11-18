import numpy as np
import cv2

''''
Return one frame

The frame is a table: (position_x, position_y, 3)
'''
def get_image():
    # Open camera (0 = first camera USB detected)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Unable to open the camera")

    # Read one frame
    ret, frame = cap.read()

    if not ret:
        raise Exception("Unable to capture the image")

    # Display the image
    cv2.imshow("Captured image", frame)

    # Attend une touche puis ferme
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release the camera
    cap.release()

    return frame


'''

'''

def get_fram_from_aruco(frame) : 

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # Create the ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)
    # Print the detected markers
    print("Detected markers:", ids)
    print("Corners:", corners)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('Detected Markers', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_homography_from_aruco(frame, aruco_dict=cv2.aruco.DICT_4X4_50):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(aruco_dict), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        raise Exception("Not enough ArUco markers detected")

    # Sort by id to keep consistent order (0,1,2,3)
    ids = ids.flatten()
    sorted_idx = np.argsort(ids)
    corners = [corners[i] for i in sorted_idx]

    # Your sheet corners in the order TL, TR, BR, BL
    dst_pts = np.array([
        [0, 0],
        [500, 0],
        [500, 700],
        [0, 700]
    ], dtype=np.float32)

    # Aruco corners: take the first corner of each marker
    src_pts = np.array([c[0][0] for c in corners], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)

    return H

'''
Correction of the image
'''
def warp_map(frame, H):
    warped = cv2.warpPerspective(frame, H, (500, 700))
    return warped


'''
Create and send the grid for the beginning (i.e. before pathfinding)

zero means black -> wall
one means white -> road
If a cell is on both -> becomes a wall
'''
def get_grid(grid_Nx,grid_Ny):
    #Get the corners -> tags
    grid = np.zeros((grid_Nx, grid_Ny), dtype=int)

    frame = get_image()

    height = frame.shape[0]
    width = frame.shape[1]

    # Compute cell size in pixels
    cell_h = height // grid_Nx
    cell_w = width  // grid_Ny

    H = get_homography_from_aruco(frame)
    hsv = cv2.cvtColor(warp_map(frame, H), cv2.COLOR_BGR2HSV)

    for i in range(grid_Nx):
        for j in range(grid_Ny):
            cell = hsv[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            h_mean = np.mean(cell[:,:,0])
            s_mean = np.mean(cell[:,:,1])
            v_mean = np.mean(cell[:,:,2])

            # ----- road (white or yellow) -----
            is_white = v_mean > 160 and s_mean < 60
            is_yellow = 15 < h_mean < 40 and s_mean > 60

            if is_white or is_yellow:
                grid[i,j] = 1    # road
            else:
                grid[i,j] = 0    # wall

    return grid


'''
Detect the robot orientation (thanks to a red line) and return it in degrees
'''
def detect_robot_orientation():

    #Get corrected image
    frame = get_image()
    H = get_homography_from_aruco(frame)
    warped = warp_map(frame, H)

    #Change that to understandable data
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Red mask for the robot direction (two ranges)
    lower1 = np.array([0, 120, 90])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 120, 90])
    upper2 = np.array([180, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )

    # Hough transform to detect line
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)

    if lines is None:
        raise Exception("Robot line not detected")

    # Choose the longest line
    line = max(lines, key=lambda L: np.hypot(L[0][2] - L[0][0], L[0][3] - L[0][1]))[0]

    x1, y1, x2, y2 = line

    angle = np.arctan2(y2 - y1, x2 - x1) * 180/np.pi

    # Robot orientation normalized to [-180,180]
    if angle < -180: angle += 360
    if angle > 180: angle -= 360

    # We can also compute robot center
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    return cx, cy, angle



#test
grid = get_grid(20, 20)
x, y, angle = detect_robot_orientation()
get_fram_from_aruco()

print(grid)
print(f"X = {x} and y = {y} and angle = {angle}")



