import numpy as np
import cv2
import matplotlib.pyplot as plt

'''

Arucos: id 0 and 2 are used for the cut (i.e. to create the map) and 
        id 1 and 3 for start (i.e. thymio) and goal position (1 for start and 3 for goal)
Grid: 0 = road, 1 = walls

Aruco size -> 5cm and max Robot size -> 12cm => cell size 
        
"Origin direction": 
in cm:
    - x axis -> right
    - y axis -> up
in pixels:
    - x axis -> right
    - y axis -> down

REMARK: Depending on your computer you could need to change the camera opening line: self.__cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

'''

class Vision:
    def __init__(self):
        #Variables for gnav/outside the class
        self.grid = None
        self.thymio_pos = None
        self.cell_size = None #cell size in cm
        self.goal = None 
        self._aruco_real_size_cm = 5  #Known ArUco physical size
        self.__cm_per_pixel_after = None 
        self.__cm_per_pixel_before = None

        #variables mainly for the class
        self.wall_dilation = 0 #0 by default -> used in get_grid()
        self.__goal_end = np.zeros((2,2)) #1 pt -> (x,y) -> NOT (y,x) (line first and column then)
        self._M, self._w, self._h = None, None, None #variables used to crop the image

        #Open camera (0 = first camera USB detected)
        self.__cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        #Doesn't allow automatic rotations
        self.__cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)   

        #Force the orientation to 0
        self.__cap.set(cv2.CAP_PROP_ORIENTATION_META, 0) 

        #Take the calibration data if it exists
        try:
            calib = np.load('camera_calibration.npz')
            self.__camera_matrix = calib['camera_matrix']
            self.__dist_coeffs = calib['dist_coeffs']
            print("Calibration chargée avec succès")
        except:
            print("ATTENTION: Pas de calibration trouvée, distortion non corrigée!")
            self.__camera_matrix = None
            self.__dist_coeffs = None

        if not self.__cap.isOpened():
            raise Exception("Unable to open the camera")
        
    def re_initialize_camera(self):
        #release and re-open the camera
        self.__cap.release()


        #Variables for gnav/outside the class
        self.grid = None 
        self.thymio_pos = None
        self.cell_size = None 
        self.goal = None
        self._aruco_real_size_cm = 5  #Known ArUco physical size
        self.__cm_per_pixel_after = None
        self.__cm_per_pixel_before = None

        #variables mainly for the class
        self.wall_dilation = 0 #0 by default -> used in get_grid()

        # Open camera (0 = first camera USB detected)
        self.__cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.__goal_end = np.zeros((2,2)) #1 pt -> (x,y) -> NOT (y,x) (line first and column then)

        #variables used to crop the image
        self._M, self._w, self._h = None, None, None

    def __del__(self):
        #Release the camera
        self.__cap.release()

    def set_wall_dilation(self,dilation):
        self.wall_dilation = dilation

    ''''
    Return one frame (filtered) from the camera
    The frame is a table: (length_x, length_y, 3)
    '''
    def get_image(self,cap,plot=False):

        # Read one frame
        ret, frame = cap.read()

        if not ret:
            raise Exception("Unable to capture the image")

        if plot:
            #Display the image
            cv2.imshow("Captured image", frame)
            #Wait for a key press then close
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #The advantage of the bilateral filter is that it manages to smooth the image all the while
        #conserving the edges. There is of course a tradeoff in terms of computation time but since 
        #we have time to do it it is ok (i.e. sampling time not too short)
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75) #filtering

        if plot:
            #Display the image
            cv2.imshow("Filtered image", frame)
            #Wait for a key press then close
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("frame.shape:", frame.shape)

        return frame
    
    '''
    Used to setup the camera -> if space key pressed -> exit the loop
    '''
    def cam_centering(self):
        while True:
            ret, frame = self.__cap.read()
            cv2.imshow("Cam centering", frame)

            key = cv2.waitKey(0)

            if key == 32:  #32 = ASCII code for space key
                break 

        cv2.destroyAllWindows()

    def plot_grid(self,half_size=0):
        print(f"Grid sizeY = {self.grid.shape[0]}, grid sizeX = {self.grid.shape[1]}")
        #Reconstruction to visualize the grid
        image_cropped = np.zeros((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 0:
                    image_cropped[i, j] = [0, 0, 0]        #black
                elif self.grid[i, j] == 1:
                    image_cropped[i, j] = [255, 255, 255]  #white
                else:
                    image_cropped[i, j] = [255, 0, 0]      #red if unknown
        
        #Do a square on goal/start
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                #Since positions are [row, col] -> row == y, col == x
                ny = int(self.thymio_pos[0]) + dy
                nx = int(self.thymio_pos[1]) + dx
                ny2 = int(self.goal[0]) + dy
                nx2 = int(self.goal[1]) + dx

                #Check that we stay within image bounds
                if 0 <= ny < image_cropped.shape[0] and 0 <= nx < image_cropped.shape[1]:
                    image_cropped[ny, nx] = [0, 255, 0]  # green (start)
                if 0 <= ny2 < image_cropped.shape[0] and 0 <= nx2 < image_cropped.shape[1]:
                    image_cropped[ny2, nx2] = [0, 0, 255]  # blue (goal)
        
        #Plot the grid
        plt.figure(figsize=(6, 6))
        plt.imshow(image_cropped, origin='upper')
        #Add grid lines (i.e. cell borders)
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, self.grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linewidth=0.5)
        ax.set_xticks(np.arange(0, self.grid.shape[1], 1), minor=False)
        ax.set_yticks(np.arange(0, self.grid.shape[0], 1), minor=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.show()

    '''
    General function
    Take a picture, cut it inside the Arucos and do the grid (0 -> road, 1 -> walls, 2 -> start, 3 -> goal)
    '''
    def vision(self,acquisition_delay,white_threshold,plot,P):
        #Camera auto-tune
        for idx in range(acquisition_delay):
            _,_ = self.__cap.read()

        frame = self.get_image(self.__cap,plot)

        #Cut the frame with the aruco (to keep only the interesting zone)
        frame_cropped,self.__goal_end = self.get_frame_from_aruco(frame)

        if plot:
            cv2.imshow("Aruco cropped", frame_cropped)
            #Wait for a key press then close
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.grid = self.get_grid(50,30,frame_cropped,white_threshold,P)#the threshold parameter can be used to tune it (in function of the workplace)

        grid_Ny, grid_Nx = self.grid.shape
        height, width = frame_cropped.shape[:2]

        #Remap start/end in the grid
        start_end_grid = []
        i = 0
        for pt in self.__goal_end:
            i += 1
            x_grid = int(pt[0] * grid_Nx)#(normalization already done in aruco cutting)
            y_grid = int(pt[1] * grid_Ny)
            #Clamp indices
            x_grid = min(max(x_grid, 0), grid_Nx-1)
            y_grid = min(max(y_grid, 0), grid_Ny-1)

            if i==1:
                #Store positions in (row, col) == (y, x) to match numpy indexing
                self.thymio_pos = [y_grid, x_grid]
            elif i==2:
                #Store positions in (row, col) == (y, x)
                self.goal = [y_grid, x_grid]
        

    '''
    Detect the Arucos and crop the frame by creating a rectangle with the center of the Arucos being the corners
    '''
    #Function de https://www.geeksforgeeks.org/computer-vision/detecting-aruco-markers-with-opencv-and-python-1/
    def get_frame_from_aruco(self,frame,plot=False):

        #Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        #Create the ArUco detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        #Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)
        if plot:
            #Print the detected markers
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
        aruco_pixel_sizes = []

        centers_2pts=np.zeros((2,2))
        id_to_idx = {0: 0, 2: 1} #since i can go to n (n being number of arucos)
        for c, i in zip(corners, ids):
            pts = c[0]  # shape (4,2)

            #Distance between top-left and top-right corners
            pixel_width = np.linalg.norm(pts[0] - pts[1])
            pixel_height = np.linalg.norm(pts[1] - pts[2])
            aruco_pixel_sizes.append((pixel_width + pixel_height) / 2)

            if i in [0, 2]:
                center = c[0].mean(axis=0)
                centers_2pts[id_to_idx[int(i)]] = center

        #Average pixel size of all detected ArUcos
        aruco_pixel_size_before = np.mean(aruco_pixel_sizes)

        #Conversion ratio
        self.__cm_per_pixel_before = self._aruco_real_size_cm/aruco_pixel_size_before

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

        #Recalculate the new pixel size in cm (i.e. the cropped image pixel size)
        #Detect the markers
        #Recalculate the new pixel size in cm on the CROPPED image (more accurate)
        gray_cropped = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        corners_after, ids_after, rejected_after = detector.detectMarkers(gray_cropped)

        aruco_pixel_sizes_after = []
        if ids_after is not None:
            for c, i in zip(corners_after, ids_after):
                pts = c[0]
                pixel_width = np.linalg.norm(pts[0] - pts[1])
                pixel_height = np.linalg.norm(pts[1] - pts[2])
                #to test if distortion
                print(f"ArUco {i[0]}: largeur={pixel_width:.1f}px, hauteur={pixel_height:.1f}px")
                aruco_pixel_sizes_after.append((pixel_width + pixel_height) / 2)

        aruco_pixel_size_after = np.mean(aruco_pixel_sizes_after) if len(aruco_pixel_sizes_after) > 0 else 0.0

        if aruco_pixel_size_after > 0:
            self.__cm_per_pixel_after = self._aruco_real_size_cm / aruco_pixel_size_after
        else:
            #Fallback to before if after-detection failed
            self.__cm_per_pixel_after = self.__cm_per_pixel_before

        #Replace the arucos for start and goal by white to avoid confusion during grid creation
        aruco_margin_factor = 2.5 #How much size of the Aruco do we want to add ?
        for c, i in zip(corners_after, ids_after):
            if i in [1, 3]:
                pts = c[0]
                center = c[0].mean(axis=0)
                expanded_pts = []
                for pt in pts:
                    vec = pt - center
                    length = np.linalg.norm(vec)
                    if length > 0:
                        new_pt = center + vec * (1 + aruco_margin_factor)
                        expanded_pts.append(new_pt)
                expanded_pts = np.array(expanded_pts, dtype=np.int32)
                cv2.fillPoly(cropped_frame, [expanded_pts], (255, 255, 255))
        return cropped_frame,start_end

    '''
    Do the cut and return the cropped frame and the transformation matrix
    + applies perspective transformation to obtain a frontal/orthogonal view
    and do a borderMode to reduce distortion at the edges
    '''
    def cut_from_aruco(self,frame,centers):

        #Minimal rectangle around the centers
        rect = cv2.minAreaRect(centers)     #(center,(w,h),angle)
        box = cv2.boxPoints(rect)           #4 corners
        box = box.astype(np.float32) 

        #Calculate actual distances between corners to preserve aspect ratio
        self._w = int(np.round(np.linalg.norm(box[1] - box[0])))
        self._h = int(np.round(np.linalg.norm(box[2] - box[1])))

        if self._w == 0 or self._h == 0:
            print("Invalid rectangle dimensions for cropping.")
            return None

        dst_pts = np.array([[0,0],[self._w,0],[self._w,self._h],[0,self._h]], dtype="float32")
        self._M = cv2.getPerspectiveTransform(box.astype("float32"), dst_pts)
        #Apply borderMode to reduce distortion at the edges
        cropped_frame = cv2.warpPerspective(frame, self._M, (self._w, self._h), borderMode=cv2.BORDER_REFLECT)
        return cropped_frame,self._M

    '''
    Get a cropped frame from the arucos. Either recalculate the cropping parameters or use the previous ones
    CAUTION: If the recalculation is done -> the arucos will be replaced by white in the image (done in get_frame_from_aruco to avoid confusion during grid creation)
    Remark: if the camera or the setup moved, it is better to recalculate the cropping parameters -> small change in code needed if you want it cleaner
    '''
    def get_cutted_frame(self,plot=False,resample=False):
        if self._M is None or self._w is None or self._h is None or resample:
            # First time setup
            if plot:
                print("Setting up cropping parameters...")
            frame = self.get_image(self.__cap,plot)
            _,_ = self.get_frame_from_aruco(frame)
            cropped_frame = cv2.warpPerspective(self.get_image(self.__cap,plot), self._M, (self._w, self._h), borderMode=cv2.BORDER_REFLECT)
        else:
            cropped_frame = cv2.warpPerspective(self.get_image(self.__cap,plot), self._M, (self._w, self._h), borderMode=cv2.BORDER_REFLECT)

        if plot:
            cv2.imshow("Cropped frame from get_cutted_frame", cropped_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return cropped_frame

    def overlay_grid_on_cropped(self, frame=None, show=True, line_color=(0,255,0), start_color=(0,255,0), goal_color=(255,0,0)):
        """
        Superpose the grid on the cropped image.
        """
        if self.grid is None:
            print("No grid available to overlay. Run vision() first.")
            return None

        if frame is None:
            frame = self.get_cutted_frame(False)
            if frame is None:
                print("Unable to get cropped frame for overlay.")
                return None

        img = frame.copy()
        grid_Ny, grid_Nx = self.grid.shape
        h, w = img.shape[:2]

        #Cell size
        cell_w_f = float(w) / float(grid_Nx)
        cell_h_f = float(h) / float(grid_Ny)

        #Draw grid lines
        for j in range(grid_Nx + 1):
            x = int(round(j * cell_w_f))
            cv2.line(img, (x, 0), (x, h), line_color, 1)
        for i in range(grid_Ny + 1):
            y = int(round(i * cell_h_f))
            cv2.line(img, (0, y), (w, y), line_color, 1)

        #Mark start and goal cell centers if available (self.thymio_pos and self.goal are [row, col])
        try:
            if getattr(self, 'thymio_pos', None) is not None:
                row, col = int(self.thymio_pos[0]), int(self.thymio_pos[1])
                cx = int(round((col + 0.5) * cell_w_f))
                cy = int(round((row + 0.5) * cell_h_f))
                cv2.circle(img, (cx, cy), max(6, int(min(cell_w_f, cell_h_f) * 0.2)), start_color, -1)

            if getattr(self, 'goal', None) is not None:
                row, col = int(self.goal[0]), int(self.goal[1])
                cx = int(round((col + 0.5) * cell_w_f))
                cy = int(round((row + 0.5) * cell_h_f))
                cv2.circle(img, (cx, cy), max(6, int(min(cell_w_f, cell_h_f) * 0.2)), goal_color, -1)
        except Exception as e:
            print(f"Error while marking start/goal on overlay: {e}")

        if show:
            cv2.imshow("Grid overlay on cropped image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img

    '''
    Create and send the grid (i.e. before pathfinding)

    zero means black -> wall
    one means white -> road
    Advice: do not use the default case with 0,0 because the A* will be very long to compute

    P: it creates a square grid with cell size P (in pixels)
    '''
    def get_grid(self,grid_Nx,grid_Ny,frame,white_th,P,plot=False):
        height, width, _ = frame.shape

        #Default case
        if grid_Nx == 0 or grid_Ny == 0 or P > 0:
            if P > 0:
                if plot:
                    print("square case in grid creation")
                grid_Nx = int(width  / P)
                grid_Ny = int(height / P)
            else:
                if plot:
                    print("Default case in grid creation")
                grid_Nx = width
                grid_Ny = height

        #Compute cell size using floating point division to avoid truncation
        cell_h_f = max(float(height) / float(grid_Ny), 1.0)
        cell_w_f = max(float(width)  / float(grid_Nx), 1.0)

        # Use rounded integer boundaries when slicing the image into cells
        # but keep the floating-point cell size for metric conversions
        cell_h = max(int(round(cell_h_f)), 1)
        cell_w = max(int(round(cell_w_f)), 1)

        #calculate cell size in cm using the more accurate float cell size
        cell_w_cm = cell_w_f * self.__cm_per_pixel_after
        cell_h_cm = cell_h_f * self.__cm_per_pixel_after
        self.cell_size = (cell_w_cm + cell_h_cm) / 2.0 #average (in cm)

        if plot:
            print(f"=== DEBUG cm_per_pixel ===")
            print(f"BEFORE crop: {self.__cm_per_pixel_before:.6f} cm/px")
            print(f"AFTER crop:  {self.__cm_per_pixel_after:.6f} cm/px")
            print(f"Ratio after/before: {self.__cm_per_pixel_after/self.__cm_per_pixel_before:.3f}")
            print(f"cell_size: {self.cell_size:.3f} cm")

        grid = np.zeros((grid_Ny, grid_Nx), dtype=int)

        #Convert to grayscale for grid processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Grid processing
        for i in range(grid_Ny):
            for j in range(grid_Nx):
                #Compute integer bounds from the floating cell size (rounding limits)
                y0 = int(round(i * cell_h_f))
                y1 = int(round((i + 1) * cell_h_f))
                x0 = int(round(j * cell_w_f))
                x1 = int(round((j + 1) * cell_w_f))

                #Clamp to image bounds
                y0 = max(0, min(y0, height))
                y1 = max(0, min(y1, height))
                x0 = max(0, min(x0, width))
                x1 = max(0, min(x1, width))

                #Extract cell
                cell = gray[y0:y1, x0:x1]

                #Avoid empty cell
                if cell.size == 0:
                    grid[i, j] = 0
                    continue

                if np.mean(cell)>white_th:
                    grid[i, j] = 1   #road
                else:
                    grid[i, j] = 0   #wall

        #Dilate walls (add safety margin) (permits to tune if bad lightning when using)
        if self.wall_dilation > 0:
            #Invert grid: walls become 1, roads become 0
            inv_grid = 1 - grid

            #Create kernel
            kernel = np.ones((2*self.wall_dilation+1, 2*self.wall_dilation+1), np.uint8)

            #Dilate only walls
            inv_grid = cv2.dilate(inv_grid.astype(np.uint8), kernel, iterations=1)

            #Re-invert: walls back to 0, roads to 1
            grid = 1 - inv_grid

        return grid

    '''
    This function detects the thymio and get its position and orientation
    Detects aruco id 1 and returns its center position and orientation based on the aruco corners
    '''
    def get_thymio_pos(self, frame):
        #Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        #Create the ArUco detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        #Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is None:
            print("No aruco markers detected")
            return None, None, None

        #Find aruco with id 1 (thymio)
        thymio_idx = None
        thymio_corners = None
        
        for idx, marker_id in enumerate(ids):
            if marker_id[0] == 1:
                thymio_idx = idx
                thymio_corners = corners[idx]
                break
        
        if thymio_idx is None:
            print("Aruco id 1 (thymio) not detected")
            return None, None, None
        
        #Get the center of the aruco
        pts = thymio_corners[0]  #shape (4,2) - [top-left, top-right, bottom-right, bottom-left]
        center = pts.mean(axis=0)
        x, y = center[0], center[1]
        
        #Calculate orientation from center to top-right corner
        #pts[1] is the top-right corner
        top_right = pts[1]
        dx = top_right[0] - x
        dy = top_right[1] - y
        angle = np.arctan2(dy, dx)

        #Normalize angle to [-pi, pi]
        if angle > np.pi:
            angle -= 2 * np.pi
        if angle < -np.pi:
            angle += 2 * np.pi

        return x, y, angle
    
    def get_thymio_pos_in_cm(self, frame):
        x_px, y_px, theta = self.get_thymio_pos(frame)
        if x_px is None or y_px is None:
            return None, None, None
        
        if self.__cm_per_pixel_after is None:
            self.__cm_per_pixel_after = 1/self.get_start_pos_and_cm_to_pixel(frame)[2]

        #Convert pixel position to cm using the scale after cropping
        x_cm = x_px * self.__cm_per_pixel_after
        #Invert y-axis to have origin at bottom-left instead of top-left
        y_cm = (frame.shape[0] - y_px) * self.__cm_per_pixel_after

        return x_cm, y_cm, theta
    
    '''
    Detecte l'ArUco d'id 1 (start) dans l'image fournie et renvoie la position
    de départ en centimètres ainsi que le facteur `pixels par cm` (cm_to_pixel).
    '''
    def calculate_cm_to_pixel(self,frame):
        #Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        cm_to_pixel = None

        if ids is None:
            print("No aruco markers detected in frame in calculate_cm_to_pixel")
            return None
        #Compute mean pixel size of detected ArUco to get scale
        aruco_pixel_sizes = []
        for c, i in zip(corners, ids):
            pts = c[0]
            pixel_width = np.linalg.norm(pts[0] - pts[1])
            pixel_height = np.linalg.norm(pts[1] - pts[2])
            aruco_pixel_sizes.append((pixel_width + pixel_height) / 2)

        if len(aruco_pixel_sizes) > 0:
            mean_pixels = np.mean(aruco_pixel_sizes)
            if mean_pixels > 0:
                #Pixels per cm
                cm_to_pixel = mean_pixels / self._aruco_real_size_cm
                #Store inverse if useful elsewhere
                self.__cm_per_pixel_after = self._aruco_real_size_cm / mean_pixels

        #If we still don't have cm_to_pixel, try cached value
        if cm_to_pixel is None and self.__cm_per_pixel_after is not None:
            print("Scale cm_to_pixel not computed from detected markers, trying cached value")
            cm_to_pixel = 1.0 / self.__cm_per_pixel_after

        return cm_to_pixel

    '''
    Detects the ArUco with ID 1 (start) in the provided image and returns the
    starting position in centimetres as well as the pixels per cm factor (cm_to_pixel).
    '''
    def get_start_pos_and_cm_to_pixel(self, frame):
        #Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        cm_to_pixel = None

        if ids is None:
            print("No aruco markers detected in frame in get_start_pos_and_cm_per_pixel")
            return None, None, None, None

        else:
            #Compute mean pixel size of detected ArUco to get scale
            aruco_pixel_sizes = []
            for c, i in zip(corners, ids):
                pts = c[0]
                pixel_width = np.linalg.norm(pts[0] - pts[1])
                pixel_height = np.linalg.norm(pts[1] - pts[2])
                aruco_pixel_sizes.append((pixel_width + pixel_height) / 2)

            if len(aruco_pixel_sizes) > 0:
                mean_pixels = np.mean(aruco_pixel_sizes)
                if mean_pixels > 0:
                    #Number of pixels per cm
                    cm_to_pixel = mean_pixels / self._aruco_real_size_cm
                    #Store inverse if useful elsewhere
                    self.__cm_per_pixel_after = self._aruco_real_size_cm / mean_pixels

        #Now try to find id 1 and compute its position in cm with origin at bottom-left
        if ids is not None:
            for idx, marker_id in enumerate(ids):
                if marker_id[0] == 1:
                    pts = corners[idx][0]
                    center_px = pts.mean(axis=0)
                    x_px = float(center_px[0])
                    y_px_top = float(center_px[1])

                    #If we still don't have cm_to_pixel, try cached value
                    if cm_to_pixel is None:
                        print("Scale cm_to_pixel not computed from detected markers, trying cached value")
                        if getattr(self, '_Vision__cm_per_pixel_after', None) is not None and self._Vision__cm_per_pixel_after > 0:
                            cm_to_pixel = 1.0 / self._Vision__cm_per_pixel_after
                        else:
                            print("Scale unavailable to convert pixels to cm")
                            return None, None, None

                    #Convert pixels to cm. x: left->right, y: bottom->top
                    x_cm = x_px / cm_to_pixel
                    y_cm = (frame.shape[0] - y_px_top) / cm_to_pixel

                    #Compute orientation of the marker (center -> top-right)
                    top_right = pts[1]
                    dx = top_right[0] - center_px[0]
                    dy = top_right[1] - center_px[1]
                    orientation = np.arctan2(dy, dx)  # radians

                    return x_cm, y_cm, cm_to_pixel, orientation

        #If we reach here, id 1 not detected
        print("Aruco id 1 (start) not detected in frame")
        #Return None for position and orientation but cm_to_pixel if available
        return None, None, cm_to_pixel, None

#test

'''
v = Vision()
#v.vision_test(5,90)
v.cam_centering()
v.vision(5,90,True,1)
#v.vision(5,150,True)

for idx in range(10):
    print(f"Test number {idx+1}")
    img = v.get_image(v._Vision__cap, False)
    cv2.imshow("Debug", img)
    x,y,theta = v.get_thymio_pos_in_cm(v.get_image(v._Vision__cap,False))
    print(f"Thymio position: x={x}, y={y}, theta={theta*180/np.pi} degrees")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    v.cam_centering()  

#v.plot_grid()

 
'''
