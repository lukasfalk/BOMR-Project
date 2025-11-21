import numpy as np
import cv2
import matplotlib.pyplot as plt

''''
Return one frame

The frame is a table: (position_x, position_y, 3)
'''
def get_image(cap):

    # Read one frame
    ret, frame = cap.read()

    if not ret:
        raise Exception("Unable to capture the image")

    # Display the image
    cv2.imshow("Captured image", frame)

    # Attend une touche puis ferme
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #The advantage of the bilateral filter is that it manages to smooth the image all the while
    #conserving the edges. There is of course a tradeoff in terms of computation time but since 
    #we have time to do it it is ok (i.e. sampling time not too short)
    frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75) #filtering
    # Display the image
    cv2.imshow("Filtered image", frame)
    # Attend une touche puis ferme
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(frame.shape)

    return frame

import time

def vision_loop(acquisition_delay):
    # Open camera (0 = first camera USB detected)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise Exception("Unable to open the camera")
    
    '''
    for w, h in [(640, 480), (1280, 720), (1920, 1080)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        time.sleep(0.2)   # important

        ret, frame = cap.read()
        print(f"Requested: {w}x{h} -> Got: {frame.shape[1]}x{frame.shape[0]}")'''
    
    #set de resolution
    '''
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)  # 3500–6500 selon l'éclairage
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # dépend de la caméra
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # à ajuster
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    time.sleep(0.5)
    '''
    #Camera auto-tune
    for idx in range(acquisition_delay):
        _,_ = cap.read()
    
    #Get the "real" frame
    frame = get_image(cap)
    
    #Cut the frame with the aruco (to keep only the interesting zone)
    

    #create grid and detect start position
    grid = get_grid(500,700,frame)
    x, y, angle = detect_robot_orientation(frame)

    #Tests 
    #print(grid)
    # Création d'une image RGB vide
    image = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

    # Attribution des couleurs
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:
                image[i, j] = [0, 0, 0]        # noir
            elif grid[i, j] == 1:
                image[i, j] = [255, 255, 255]  # blanc
            elif grid[i, j] == 2:
                image[i, j] = [255, 255, 0]    # jaune
            else:
                image[i, j] = [0, 0, 255]      # bleu pour tout autre

    # Affichage de l'image
    plt.imshow(image[::-1])
    plt.axis('off')  # enlever les axes
    plt.show()

    print(f"X = {x} and y = {y} and angle = {angle}")

    get_fram_from_aruco(frame)

    
    # Release the camera
    cap.release()

'''

'''
# Function de https://www.geeksforgeeks.org/computer-vision/detecting-aruco-markers-with-opencv-and-python-1/
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
def get_grid(grid_Nx,grid_Ny,frame):

    height, width, _ = frame.shape

    # Compute cell size in pixels
    if(grid_Nx==0 or grid_Ny == 0):
        #default case
        print("Default case in grid creation")
        grid_Nx = width
        grid_Ny = height

    cell_h = max(height // grid_Ny, 1)
    cell_w = max(width  // grid_Nx, 1)

    grid = np.zeros((grid_Ny, grid_Nx), dtype=int)

    '''
    H = get_homography_from_aruco(frame)
    hsv = cv2.cvtColor(warp_map(frame, H), cv2.COLOR_BGR2HSV)'''

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for i in range(grid_Ny):
        for j in range(grid_Nx):
            '''
            cell = hsv[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            h_mean = np.mean(cell[:,:,0])
            s_mean = np.mean(cell[:,:,1])
            v_mean = np.mean(cell[:,:,2])

            # ----- road (white or yellow) -----
            #is_white = v_mean > 90 and s_mean < 150 #trial and error
            is_white = 0
            is_yellow = 15 <= h_mean <= 60 and s_mean > 80 and v_mean > 80

            #if is_white or is_yellow:
            if is_white:
                grid[i,j] = 1    # road
            elif is_yellow:
                grid[i,j] = 2    # crosswalk
            else:
                grid[i,j] = 0    # wall
                '''
            
            # Extraire la cellule
            cell_rgb = frame[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

            # Moyenne par canal
            r_mean = np.mean(cell_rgb[:,:,2])
            g_mean = np.mean(cell_rgb[:,:,1])
            b_mean = np.mean(cell_rgb[:,:,0])

            # Détection blanc
            is_white = r_mean > 100 and g_mean > 100 and b_mean > 100

            # Détection jaune
            is_yellow = r_mean > 100 and g_mean > 80 and b_mean < 90

            if is_yellow:
                grid[i,j] = 2    # crosswalk
            elif is_white:
                grid[i,j] = 1    # road
            else:
                grid[i,j] = 0    # wall

    return grid

'''
This function detects the thymio and get its position with the (0,0) at the bottom left aruco
'''
def get_thymio_pos():
    pos = [0,0]
    return pos



'''
Detect the robot orientation (thanks to a red line) and return it in degrees
'''
def detect_robot_orientation(frame):

    #Get corrected image
    '''
    H = get_homography_from_aruco(frame) #si working -> créer un H dans la fonction loop !!!!!
    warped = warp_map(frame, H)

    #Change that to understandable data
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    '''

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
        print("Robot line not detected")
        return -1,-1,-1

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
'''
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    cv2.imshow("raw", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC pour quitter
        break
cap.release()
cv2.destroyAllWindows()'''

vision_loop(20)



