import numpy as np
import cv2
import matplotlib.pyplot as plt
import time #to see if too slow (can be deleted)



class Vision:
    def __init__(self):
        '''
        self.cap = None        # public
        self._grid = None      # protected
        self.__secret = 42     # private
        '''
        # Open camera (0 = first camera USB detected)
        self.__cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.__goal_end = np.zeros((2,2))

        if not self.__cap.isOpened():
            raise Exception("Unable to open the camera")

    def __del__(self):
        #Release the camera
        self.__cap.release()

    ''''
    Return one frame

    The frame is a table: (length_x, length_y, 3)
    '''
    def get_image(self,cap,plot):

        # Read one frame
        ret, frame = cap.read()

        if not ret:
            raise Exception("Unable to capture the image")

        if plot:
            # Display the image
            cv2.imshow("Captured image", frame)
            # Attend une touche puis ferme
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #The advantage of the bilateral filter is that it manages to smooth the image all the while
        #conserving the edges. There is of course a tradeoff in terms of computation time but since 
        #we have time to do it it is ok (i.e. sampling time not too short)
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75) #filtering

        if plot:
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

    def acquisition_loop(self):
        plot = False  # affichage activé par défaut

        print("Appuie sur :")
        print(" - ESPACE pour activer/désactiver l'affichage de l'image")
        print(" - ESC pour quitter")

        while True:
            try:
                acquisition_delay = int(input("Entre acquisition_delay : "))
            except:
                print("Valeur invalide.")
                continue

            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            start = time.time()

            # Camera auto-tune
            for _ in range(acquisition_delay):
                cap.read()

            # Get the "real" frame
            frame = self.get_image(cap, plot)
            
            end = time.time()
            print(f"Timing acquisition = {end - start:.6f} s")

            # Plot seulement si activé
            if frame is not None:
                cv2.imshow("Frame", frame)

            key = cv2.waitKey(1) & 0xFF
            cap.release()

            # ESC -> stop
            if key == 27:
                print("Arrêt.")
                break
            
            # ESPACE -> toggle plot
            elif key == 32:
                plot = not plot
                print("Affichage image :", plot)
            

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
        #to kick
        #self.__cap.release()
        #self.acquisition_loop()
        #self.cam_centering(self.__cap)

        plot = 0
        start = time.time()
        #Camera auto-tune
        for idx in range(acquisition_delay):
            _,_ = self.__cap.read()
        
        #Get the "real" frame
        frame = self.get_image(self.__cap,plot)
        end = time.time()
        print(f"timing of getting an image (with the auto-tune) = {end - start}")#time of 684ms with acquisition_delay of 20 and 160ms to 210ms with acquisition_delay = 5 (and image sufficiently good)
        
        start = time.time()
        #Cut the frame with the aruco (to keep only the interesting zone)
        frame_cropped,self.__goal_end = self.get_frame_from_aruco(frame)
        end = time.time()
        print(f"timing of cropping = {end - start}")
        #Display the image
        cv2.imshow("Aruco cropped", frame_cropped)
        #Attend une touche puis ferme
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        start = time.time()
        grid = self.get_grid(0,0,frame_cropped,150)#the last threshold parameter can be used to tune it (in function of the workplace)
        end = time.time()
        print(f"timing of get grid = {end - start}")
        start = time.time()
        x, y, angle = self.detect_robot_orientation(frame_cropped)
        end = time.time()
        print(f"X = {x} and Y = {y} and angle = {angle}")
        print(f"Timing of getting an angle = {end - start}")


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

        grid_Ny, grid_Nx = grid.shape
        height, width = frame_cropped.shape[:2]

        # Remap start/end dans la grille
        start_end_grid = []
        for pt in self.__goal_end:
            print(f"x = {pt[0]} and y = {pt[1]}")
            x_norm = pt[0] #(normalization already done)
            y_norm = pt[1]
            print(f"x_norm = {x_norm} and y_norm = {y_norm}")
            print(f"grid_Nx = {grid_Nx} and grid_Ny = {grid_Ny}")
            x_grid = int(x_norm * grid_Nx)
            y_grid = int(y_norm * grid_Ny)
            # Clamp indices
            x_grid = min(max(x_grid, 0), grid_Nx-1)
            y_grid = min(max(y_grid, 0), grid_Ny-1)
            start_end_grid.append((y_grid, x_grid))
            print(f"x_grid = {x_grid} and y_grid = {y_grid}")
            zone_size = 5 #zone to print
            half_size = zone_size // 2

            # Boucle pour chaque pixel dans la zone
            for dy in range(-half_size, half_size + 1):
                for dx in range(-half_size, half_size + 1):
                    ny = y_grid + dy
                    nx = x_grid + dx
                    
                    # Vérifie que l'on reste dans les limites de l'image
                    if 0 <= ny < image_cropped.shape[0] and 0 <= nx < image_cropped.shape[1]:
                        image_cropped[ny, nx] = [0, 255, 0]
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
    def get_frame_from_aruco(self,frame) : 

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
        frame_aruco = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_aruco, corners, ids)
            cv2.imshow('Detected Markers', frame_aruco)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        '''V1
        centers = []
        for c in corners:
            pts = c[0]                      # shape (4,2)
            center = pts.mean(axis=0)       # centre (x,y)
            centers.append(center)

        centers = np.array(centers, dtype=np.float32)
        cropped_frame = self.cut_from_aruco(frame,centers)
        '''

        #id 0 is top left and id 2 is bottom right
        #Detection of the "box"
        centers_2pts=np.zeros((2,2))
        id_to_idx = {0: 0, 2: 1} #since i can go to n (n being number of arucos)
        aruco_margin = 5
        for c, i in zip(corners, ids):
            if i in [0, 2]:
                center = c[0].mean(axis=0)
                centers_2pts[id_to_idx[int(i)]] = center
                #centers_2pts[i] = center
            elif i in [1,3]:
                pts = c[0].astype(int)

                x_min = pts[:, 0].min()-aruco_margin
                x_max = pts[:, 0].max()+aruco_margin
                y_min = pts[:, 1].min()-aruco_margin
                y_max = pts[:, 1].max()+aruco_margin

                frame[y_min:y_max, x_min:x_max] = 255  #whiting filling of the goal/strat arucos (with a margin)

        top_right = [centers_2pts[0][0],centers_2pts[1][1]]
        bottom_left = [centers_2pts[1][0],centers_2pts[0][1]]

        #[top-left, top-right, bottom-right, bottom-left]
        centers = np.array([centers_2pts[0], top_right, centers_2pts[1], bottom_left], dtype=np.float32)

        centers = np.array(centers, dtype=np.float32)
        cropped_frame,M = self.cut_from_aruco(frame,centers)

        start_end=np.zeros((2,2))
        id_to_idx = {1: 0, 3: 1} #since i can go to n (n being number of arucos)
        for c, i in zip(corners, ids):
            if i in [1, 3]:
                center = c[0].mean(axis=0)
                start_end[id_to_idx[int(i)]] = center

        #Normalize
        for i in range(2):
            pt = np.array([[start_end[i]]], dtype=np.float32)  # (1,1,2)
            projected = cv2.perspectiveTransform(pt, M)#Aplly the cropped transformation on the goal/start centers
            start_end[i] = projected[0][0]
 
            # Normalisation dans le repère DU CROP
            start_end[i][0] /= cropped_frame.shape[1]  # largeur
            start_end[i][1] /= cropped_frame.shape[0]  # hauteur
        return cropped_frame,start_end

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
        return cropped_frame,M


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

        # Optional (do we put it ??): dilate walls only (add safety margin) (permits to tune if bad lightning when using)
        wall_dilation = 0 #to tune
        if wall_dilation > 0:
            # Invert grid: walls become 1, roads become 0
            inv_grid = 1 - grid

            # Create kernel
            kernel = np.ones((2*wall_dilation+1, 2*wall_dilation+1), np.uint8)

            # Dilate only walls
            inv_grid = cv2.dilate(inv_grid.astype(np.uint8), kernel, iterations=1)

            # Re-invert: walls back to 0, roads to 1
            grid = 1 - inv_grid

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
v.vision_test(5)



