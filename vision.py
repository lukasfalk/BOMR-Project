import numpy as np
import cv2
import matplotlib.pyplot as plt



class Vision:
    def __init__(self):
        '''
        self.cap = None        # public
        self._grid = None      # protected
        self.__secret = 42     # private
        '''
        # Open camera (0 = first camera USB detected)
        self.__cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        if not self.__cap.isOpened():
            raise Exception("Unable to open the camera")

    def __del__(self):
        #Release the camera
        self.__cap.release()

    ''''
    Return one frame

    The frame is a table: (length_x, length_y, 3)
    '''
    def get_image(self,cap):

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

    '''
    Used to setup the camera -> if space key pressed -> exit the loop
    '''
    def cam_centering(self,cap):
        while True:
            ret, frame = cap.read()
            cv2.imshow("Cam centering", frame)

            key = cv2.waitKey(0)

            if key == 32:  # 32 = code ASCII de la barre d'espace
                break 

        cv2.destroyAllWindows()

    def vision_test(self,acquisition_delay):        
        '''
        for w, h in [(640, 480), (1280, 720), (1920, 1080)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            time.sleep(0.2)   # important

            ret, frame = cap.read()
            print(f"Requested: {w}x{h} -> Got: {frame.shape[1]}x{frame.shape[0]}")'''

        #permit us to move the camera and find the good spot
        self.cam_centering(self.__cap)

        #Camera auto-tune
        for idx in range(acquisition_delay):
            _,_ = self.__cap.read()
        
        #Get the "real" frame
        frame = self.get_image(self.__cap)
        
        #Cut the frame with the aruco (to keep only the interesting zone)
        frame_cropped = self.get_fram_from_aruco(frame)
        #Display the image
        cv2.imshow("Aruco cropped", frame_cropped)
        #Attend une touche puis ferme
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        grid = self.get_grid(0,0,frame_cropped,90)
        x, y, angle = self.detect_robot_orientation(frame_cropped)
        print(f"X = {x} and Y = {y} and angle = {angle}")


        #Reconstruction to visualize the grid
        image_cropped = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 0:
                    image_cropped[i, j] = [0, 0, 0]        # noir
                elif grid[i, j] == 1:
                    image_cropped[i, j] = [255, 255, 255]  # blanc
                else:
                    image_cropped[i, j] = [0, 0, 255]      # bleu pour tout autre

        #Plot the grid
        plt.imshow(image_cropped, origin='upper')
        plt.axis('off')
        plt.show()

        #plot the detected red line on the grid

        # Définir la ligne rouge pour le robot
        length = 5  # longueur de la ligne en pixels
        end_x = x + length * np.cos(np.deg2rad(angle))
        end_y = y + length * np.sin(np.deg2rad(angle))

        # Affichage
        plt.imshow(image_cropped, origin='upper')
        plt.plot([x, end_x], [y, end_y], color='red', linewidth=2)
        plt.axis('off')
        plt.show()

    '''
    Detect the Arucos and crop the frame by creating a rectangle with the center of the Arucos being the corners
    '''
    # Function de https://www.geeksforgeeks.org/computer-vision/detecting-aruco-markers-with-opencv-and-python-1/
    def get_fram_from_aruco(self,frame) : 

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

        centers = []
        for c in corners:
            pts = c[0]                      # shape (4,2)
            center = pts.mean(axis=0)       # centre (x,y)
            centers.append(center)

        centers = np.array(centers, dtype=np.float32)
        cropped_frame = self.cut_from_aruco(frame,centers)
        return cropped_frame

    def cut_from_aruco(self,frame,centers):

        if(len(centers)<4):
            print("Less than 4 Arucos detected -> still try")

        # --- Calcul du rectangle minimal ---
        rect = cv2.minAreaRect(centers)     # (center,(w,h),angle)
        box = cv2.boxPoints(rect)           # 4 coins du rectangle
        box = box.astype(np.int32) 

        # --- Découpe en warp ---
        w = int(rect[1][0])
        h = int(rect[1][1])

        if w == 0 or h == 0:
            print("Rectangle invalide.")
            return None

        dst_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype="float32")
        M = cv2.getPerspectiveTransform(box.astype("float32"), dst_pts)
        cropped_frame = cv2.warpPerspective(frame, M, (w, h))
        return cropped_frame


    '''
    Create and send the grid (i.e. before pathfinding)

    zero means black -> wall
    one means white -> road
    Advice: use the default case with 0,0
    '''
    def get_grid(self,grid_Nx,grid_Ny,frame,white_th):
        height, width, _ = frame.shape

        #Default case
        if grid_Nx == 0 or grid_Ny == 0:
            print("Default case in grid creation")
            grid_Nx = width
            grid_Ny = height

        #Compute cell size (avoid zero-sized windows)
        cell_h = max(height // grid_Ny, 1)
        cell_w = max(width  // grid_Nx, 1)

        grid = np.zeros((grid_Ny, grid_Nx), dtype=int)

        #Filter red and yellow first -> to be more robust when creating the grid (and it is not too bad if grid creation takes slightly more time)
        #Convert image to filter
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #Red mask
        lower1 = np.array([0, 120, 90])
        upper1 = np.array([10, 255, 255])

        lower2 = np.array([170, 120, 90])
        upper2 = np.array([180, 255, 255])

        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )

        #Yellow mask
        lower_yellow = np.array([20, 120, 90])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        #Combine red and yellow masks
        mask_red_yellow = cv2.bitwise_or(mask_red, mask_yellow)

        #Apply mask: make red and yellow areas white
        frame_filtered = frame.copy()
        frame_filtered[mask_red_yellow > 0] = [255, 255, 255]

        #Convert to grayscale for grid processing
        gray = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)

        #grid processing
        for i in range(grid_Ny):
            for j in range(grid_Nx):
                #Extract cell
                cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

                #Avoid empty cell warning
                if cell.size == 0:
                    grid[i, j] = 0
                    continue

                if cell>white_th:
                    grid[i, j] = 1   #road
                else:
                    grid[i, j] = 0   #wall
        return grid

    '''
    This function detects the thymio and get its position with the (0,0) at the bottom left aruco
    '''
    def get_thymio_pos(self,frame):
        #to use if detect_robot_orientation not sufficient
        pos = [0,0]
        return pos

    '''
    Detect the robot orientation (thanks to a red line) and return it in degrees
    '''
    def detect_robot_orientation(self,frame):
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
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=5)


        #Apply mask red
        frame_filtered = frame.copy()
        frame_filtered[mask > 0] = [0, 0, 255] #RGB blue
        '''
        #visualize
        cv2.imshow('Red filter', frame_filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        if lines is None:
            print("Robot line not detected")
            return -1,-1,-1

        #Choose the longest line to avoid small color errors
        line = max(lines, key=lambda L: np.hypot(L[0][2] - L[0][0], L[0][3] - L[0][1]))[0]

        x1, y1, x2, y2 = line

        angle = np.arctan2(y2 - y1, x2 - x1) * 180/np.pi

        #Robot orientation normalized to [-180,180]
        if angle < -180: angle += 360
        if angle > 180: angle -= 360

        #compute robot center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        return cx, cy, angle

#test
v = Vision()
v.vision_test(20)



