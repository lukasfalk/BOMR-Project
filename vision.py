import numpy as np
import cv2
import matplotlib.pyplot as plt
import time #to see if too slow (can be deleted)


'''

Arucos: id 0 and 2 are used for the cut (i.e. to create the map) and 
        id 1 and 3 for start (i.e. thymio) and goal position (1 for start and 3 for goal)
Grid: 0 = road, 1 = walls, 2 = start, 3 = goal

Aruco size -> 5cm and max Robot size -> 12cm => cell size 
        
'''

class Vision:
    def __init__(self):
        '''
        self.cap = None        # public
        self._grid = None      # protected
        self.__secret = 42     # private
        '''
        #Variables for Vision
        self.grid = None 
        self.thymio_pos = None
        self.cell_size = None 
        self.goal = None
        self.__cm_per_pixel_after = None
        self.__cm_per_pixel_before = None

        # Open camera (0 = first camera USB detected)
        self.__cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.__goal_end = np.zeros((2,2)) #1 pt -> (x,y) -> NOT (y,x) (line first and column then)

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
    def cam_centering(self):
        while True:
            ret, frame = self.__cap.read()
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

    def plot_grid(self):
        print(f"Grid sizeY = {self.grid.shape[0]}, grid sizeX = {self.grid.shape[1]}")
        #Reconstruction to visualize the grid
        image_cropped = np.zeros((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 0:
                    image_cropped[i, j] = [0, 0, 0]        # noir
                elif self.grid[i, j] == 1:
                    image_cropped[i, j] = [255, 255, 255]  # blanc
                elif self.grid[i,j] == 2:
                    image_cropped[i, j] = [0, 255, 0]  # green (start)
                elif self.grid[i,j] == 3:
                    image_cropped[i, j] = [0, 0, 255]  # blue (goal)
                else:
                    image_cropped[i, j] = [255, 0, 0]      # red if unknown
        
        # Do a big square on goal/start
        half_size=2
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                ny = int(self.thymio_pos[1]) + dy
                nx = int(self.thymio_pos[0]) + dx
                ny2 = int(self.goal[1]) + dy
                nx2 = int(self.goal[0]) + dx
                
                # Vérifie que l'on reste dans les limites de l'image
                if 0 <= ny < image_cropped.shape[0] and 0 <= nx < image_cropped.shape[1]:
                    image_cropped[ny, nx] = [0, 255, 0]  # green (start)
                if 0 <= ny2 < image_cropped.shape[0] and 0 <= nx2 < image_cropped.shape[1]:
                    image_cropped[ny2, nx2] = [0, 0, 255]  # blue (goal)
        
        #Plot the grid
        plt.imshow(image_cropped, origin='upper')
        plt.axis('off')
        plt.show()

    '''
    General function
    Take a picture, cut it inside the Arucos and do the grid (0 -> road, 1 -> walls, 2 -> start, 3 -> goal)
    '''
    def vision(self,acquisition_delay,white_threshold,plot):
        #Camera auto-tune
        for idx in range(acquisition_delay):
            _,_ = self.__cap.read()

        frame = self.get_image(self.__cap,plot)

        #Cut the frame with the aruco (to keep only the interesting zone)
        frame_cropped,self.__goal_end = self.get_frame_from_aruco(frame)

        if plot:
            cv2.imshow("Aruco cropped", frame_cropped)
            #Attend une touche puis ferme
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.grid = self.get_grid(0,0,frame_cropped,white_threshold)#the last threshold parameter can be used to tune it (in function of the workplace)

        grid_Ny, grid_Nx = self.grid.shape
        height, width = frame_cropped.shape[:2]

        # Remap start/end in the grid
        start_end_grid = []
        i = 0
        for pt in self.__goal_end:
            i += 1
            x_grid = int(pt[0] * grid_Nx)#(normalization already done in aruco cutting)
            y_grid = int(pt[1] * grid_Ny)
            # Clamp indices
            x_grid = min(max(x_grid, 0), grid_Nx-1)
            y_grid = min(max(y_grid, 0), grid_Ny-1)

            self.grid[x_grid,y_grid] = i+2# start => 2 and goal => 3 #to plot it
            if i==1:
                self.thymio_pos = [x_grid,y_grid]
            elif i==2:
                self.goal = [x_grid,y_grid]
        

    def vision_test(self,acquisition_delay,white_threshold):        
        '''
        for w, h in [(640, 480), (1280, 720), (1920, 1080)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            time.sleep(0.2)   # important

            ret, frame = cap.read()
            print(f"Requested: {w}x{h} -> Got: {frame.shape[1]}x{frame.shape[0]}")'''

        #permit us to move the camera and find the good spot
        self.cam_centering()
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
        grid = self.get_grid(0,0,frame_cropped,white_threshold)#the last threshold parameter can be used to tune it (in function of the workplace)
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

        #id 0 is top left and id 2 is bottom right
        #Detection of the "box" + calculation of the cell size
        aruco_real_size_cm = 5.0  # Your known ArUco physical size
        aruco_pixel_sizes = []

        centers_2pts=np.zeros((2,2))
        id_to_idx = {0: 0, 2: 1} #since i can go to n (n being number of arucos)
        aruco_margin_factor = 0.1 #how much size of the Aruco do we want to add ?
        for c, i in zip(corners, ids):
            pts = c[0]  # shape (4,2)

            # Distance between top-left and top-right corners
            pixel_width = np.linalg.norm(pts[0] - pts[1])
            pixel_height = np.linalg.norm(pts[1] - pts[2])
            aruco_pixel_sizes.append((pixel_width + pixel_height) / 2)

            if i in [0, 2]:
                center = c[0].mean(axis=0)
                centers_2pts[id_to_idx[int(i)]] = center
                #centers_2pts[i] = center
                
            elif i in [1,3]:#replace the two arucos which are inside by white squares (to avoid confusion during the grid creation)
                center = c[0].mean(axis=0)

                expanded_pts = []
                for pt in pts:#radially expand it
                    vec = pt - center
                    length = np.linalg.norm(vec)
                    if length > 0:
                        new_pt = center + vec * (1 + aruco_margin_factor)
                        expanded_pts.append(new_pt)

                expanded_pts = np.array(expanded_pts, dtype=np.int32)
                cv2.fillPoly(frame, [expanded_pts], (255, 255, 255))

        # Average pixel size of all detected ArUcos
        aruco_pixel_size_before = np.mean(aruco_pixel_sizes)

        # Conversion ratio
        self.__cm_per_pixel_before = 5.0/aruco_pixel_size_before

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
            start_end[i][0] /= cropped_frame.shape[1]  # width (shape[1] = columns)
            start_end[i][1] /= cropped_frame.shape[0]  # height (shape[0] = rows)

        # Recalculate the new pixel size in cm (i.e. the cropped image pixel size)
        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)
        # Print the detected markers to see if the croped markers are detected
        print("Detected markers:", ids)
        print("Corners:", corners)
        frame_aruco = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_aruco, corners, ids)
            cv2.imshow('Detected Markers', frame_aruco)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        aruco_pixel_sizes = []

        for c, i in zip(corners, ids):
            pts = c[0]  # shape (4,2)

            # Distance between top-left and top-right corners
            pixel_width = np.linalg.norm(pts[0] - pts[1])
            pixel_height = np.linalg.norm(pts[1] - pts[2])
            aruco_pixel_sizes.append((pixel_width + pixel_height) / 2)

        # Average pixel size of all detected ArUcos
        aruco_pixel_size_after = np.mean(aruco_pixel_sizes)

        # Conversion ratio
        self.__cm_per_pixel_after = 5.0/aruco_pixel_size_after
        '''
        #test cm_per_pixel_after/before by creating a line of 40 cm and 1 cm of width
        # Ligne sur image non croppée
        cv2.line(frame, (10, 10), (10 + int(5 / self.__cm_per_pixel_before), 10), (0, 255, 0), 2)
        cv2.imshow("Avant crop", frame)
        cv2.waitKey(0)

        # Ligne sur image croppée
        cv2.line(cropped_frame, (10, 10), (10 + int(5 / self.__cm_per_pixel_after), 10), (0, 0, 255), 2)
        cv2.imshow("Après crop", cropped_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return cropped_frame,start_end

    '''
    Do the cut and return the cropped frame and the transformation matrix
    + applies perspective transformation to obtain a frontal/orthogonal view
    and do a borderMode to reduce distortion at the edges
    '''
    def cut_from_aruco(self,frame,centers):

        # --- Calcul du rectangle minimal ---
        rect = cv2.minAreaRect(centers)     # (center,(w,h),angle)
        box = cv2.boxPoints(rect)           # 4 coins du rectangle
        box = box.astype(np.float32) 

        # Calculate actual distances between corners to preserve aspect ratio
        w = int(np.round(np.linalg.norm(box[1] - box[0])))
        h = int(np.round(np.linalg.norm(box[2] - box[1])))

        if w == 0 or h == 0:
            print("Rectangle invalide.")
            return None

        dst_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype="float32")
        M = cv2.getPerspectiveTransform(box.astype("float32"), dst_pts)
        # Apply borderMode to reduce distortion at the edges
        cropped_frame = cv2.warpPerspective(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
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

        #calculte cell size in cm
        cell_w_cm = cell_w * self.__cm_per_pixel_after
        cell_h_cm = cell_h * self.__cm_per_pixel_after
        self.cell_size = (cell_w_cm + cell_h_cm) / 2 #average

        '''
        # Test visuel de cell_size sur l'image croppée
        print(f"cell_size (cm) = {self.cell_size}")
        # Trace une ligne de longueur 5*cell_size sur l'image croppée
        try:
            frame_copy = frame.copy()
            start_point = (20, 20)
            end_point = (20 + int(5*self.cell_size / self.__cm_per_pixel_after), 20)
            cv2.line(frame_copy, start_point, end_point, (255, 0, 255), 2)
            cv2.imshow("Test cell_size (ligne magenta)", frame_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Erreur lors du test visuel cell_size: {e}")
            '''

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
#v.vision_test(5,90)
v.cam_centering()
v.vision(5,90,True)
v.plot_grid()


