import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from heapq import heappush, heappop

CHANGING_DIR_PENALITY = 1

class GlobalNavigation : 

    def __init__(self):

        self.Start = None
        self.Goal = None
        self.grid = None
        self.cell_size = None
        self.thymio_size = None

        #Adapt thymio size for grid
        self.robot_size = None
        self.robot_size_cells = None

    def set_gnav(self, vision):

        row, col = vision.thymio_pos
        self.Start = row, col
        row, col = vision.goal
        self.Goal = row, col
        self.grid = vision.grid
        self.cell_size = vision.cell_size
        self.thymio_size = 12

        #Adapt thymio size for grid
        self.robot_size = int(np.ceil(self.thymio_size / self.cell_size)) // 2 #np.ceil to take he superior int
        self.robot_size_cells = 2 * self.robot_size + 1  # Rayon à gauche + centre + rayon à droite


    
    
    #Function to verify if the robot is inside the grid and not on an obstacle.
    def verification(self):
        row, col = self.Start
        print("Ce qu'on check à droite : ", -self.robot_size + col)
        print("Ce qu'on check à gauche : ", self.robot_size + col)
        for i in range(-self.robot_size + row, row + self.robot_size):
            for j in range(-self.robot_size + col, col + self.robot_size):
                
                if i >= len(self.grid) or j >= len(self.grid[0]):
                    print("Error : Thymio is not inside the grid !")
                elif self.grid[i][j] == 0 :
                    print("Error : Thymio is on an obstable !")

    #Function to adapt the path for thymio robot dimension
    def growing_obstacles(self):
        
        modified_grid = np.copy(self.grid)
        
        print(len(self.grid))
        print(len(self.grid[0]))

        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                
                if self.grid[i][j] == 0:  
                    
                    for di in range(-self.robot_size, self.robot_size+1):
                        for dj in range(-self.robot_size, self.robot_size+1):
                            
                            ni = i + di
                            nj = j + dj
                            if 0 <= ni < len(self.grid) and 0 <= nj < len(self.grid[0]):
                                modified_grid[ni][nj] = 0 # adding obstacles

        self.grid = modified_grid
    
    #Compute the diagonal heuristic
    def heuristic(self, a, b) : 

        D = 1
        D2 = np.sqrt(2)
        da = abs(a[0] - b[0]) 
        db = abs(a[1] - b[1])
        
        return D*(da + db) + (D2 - 2*D)*min(da,db)

    #Path finding algorithme with A* and Manhattan distance 
    def grid_search(self): 
        
        #Function to adapt the path for thymio robot dimension
        self.growing_obstacles()            #function to growth obstacle size
        self.verification()                 #Print error message if outside 

        # initialize the variables above
        came_from = {}                      # to reconstruct path
        g_costs = {self.Start: 0}           # cost from start to the cell
        explored = set()                    # to keep track of explored cells
        dir = {self.Start: (0, 0)}                            #to track the direction and add a direction penality        

        open_set = [(self.heuristic(self.Start, self.Goal), 0, self.Start)]  # priority queue for A* (f_cost, g_cost, position)
            
        while open_set: # Goal is unmarked
        
            current_f_cost, current_g_cost, current_pos = heappop(open_set) #Take the last position in grid

            explored.add(current_pos) 
            
            if current_pos == self.Goal: # if Goal is marked then stop the algorithme
                break

            # Get neighbors -> be careful it is like a matrix so we do (y, x)
            neighbors = [
                (current_pos[0]-1, current_pos[1]),   #Up
                (current_pos[0]+1, current_pos[1]),   #Down
                (current_pos[0], current_pos[1]-1),   #Left
                (current_pos[0], current_pos[1]+1),   #Right
                (current_pos[0]-1, current_pos[1]-1), #Diagonal Up-Left
                (current_pos[0]-1, current_pos[1]+1), #Diagonal Up-Right
                (current_pos[0]+1, current_pos[1]-1), #Diagonal Down-Left
                (current_pos[0]+1, current_pos[1]+1)  #Diagonal Down-Right
            ]

            for neighbor in neighbors: # for each neighbor of marked cells
                # Check if neighbor is within bounds
                if ((0 <= neighbor[0] < len(self.grid)) and 
                    (0 <= neighbor[1] < self.grid.shape[1])):
                    
                    # Check if neighbor is not an obstacle
                    if self.grid[neighbor[0], neighbor[1]] != 0 :

                        if neighbor in explored:
                            continue
                        
                        dir[neighbor] = (neighbor[0] - current_pos[0], neighbor[1] - current_pos[1])

                        #Compute cost for distance
                        if (abs(neighbor[0] - current_pos[0]) == 1 and 
                            abs(neighbor[1] - current_pos[1]) == 1) :
                            move_cost = np.sqrt(2) #Diagonal cost is sqrt(2)
                        else : 
                            move_cost = 1 #Orthogonal cost is 1 
                        
                        #Compute cost for angle penality 
                        if ((dir[current_pos] != (0, 0)) and
                            (dir[neighbor] != dir[current_pos])) : 
                            dir_cost = CHANGING_DIR_PENALITY  #Penality of 1
                        else : 
                            dir_cost = 0

                        tentative_g_cost = current_g_cost + move_cost + dir_cost

                        if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                            g_costs[neighbor] = tentative_g_cost
                            came_from[neighbor] = current_pos 
                        
                            f_cost = tentative_g_cost + self.heuristic(neighbor, self.Goal)
                            heappush(open_set, (f_cost, tentative_g_cost, neighbor))
                            
        # Reconstruct path
        if current_pos == self.Goal:
            path = []
            while current_pos != self.Start:
                path.append(current_pos)
                current_pos = came_from[current_pos]
            path.append(self.Start)
            path.reverse()
            return path, explored, None  # Return reversed path and explored cells
        else:
        # If we reach here, no path was found
            return None, explored, None
        
    def display_grid_with_path(self, path):

        # Créer une copie de la grille pour ne pas modifier l'originale
        grid_with_path = self.grid.copy()

        thickness = 2  # Épaisseur pour le chemin, start et goal
        half_thickness = thickness // 2

        # Marquer le chemin sur la grille avec épaisseur
        for row, col in path:
            for di in range(-half_thickness, half_thickness + 1):
                for dj in range(-half_thickness, half_thickness + 1):
                    ni = row + di
                    nj = col + dj
                    if 0 <= ni < len(grid_with_path) and 0 <= nj < len(grid_with_path[0]):
                        grid_with_path[ni, nj] = 2  # Utiliser '2' pour représenter le chemin
        
        #Assign start and goal for different color on map with thickness
        start_y, start_x = path[0]
        for di in range(-half_thickness, half_thickness + 1):
            for dj in range(-half_thickness, half_thickness + 1):
                ni = start_y + di
                nj = start_x + dj
                if 0 <= ni < len(grid_with_path) and 0 <= nj < len(grid_with_path[0]):
                    grid_with_path[ni, nj] = 3

        goal_y, goal_x = path[-1]
        for di in range(-half_thickness, half_thickness + 1):
            for dj in range(-half_thickness, half_thickness + 1):
                ni = goal_y + di
                nj = goal_x + dj
                if 0 <= ni < len(grid_with_path) and 0 <= nj < len(grid_with_path[0]):
                    grid_with_path[ni, nj] = 4

        self.grid = grid_with_path

   
    def display_colored_grid(self):

        # 0 = black, 1 = white, 2 = yellow, 3 = red, 4 = blue"
        cmap = ListedColormap(['black', 'white', 'yellow', 'red', 'blue']) 

    
        plt.imshow(self.grid, cmap=cmap, origin='upper', aspect='equal')

        # Add lines between cells
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, self.grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

        plt.title("Colored Grid Visualization")
        plt.xlabel("X-axis (Columns)")
        plt.ylabel("Y-axis (Rows)")
        plt.show()

    def vectors_for_displacement(self, path) : 
        
        path = np.array(path)

        #[delta_y, delta_x] = [y[i+1] - y[i], x[i+1] - x[i]]
        delta = np.diff(path, axis = 0)

        #Norm between two cells in the center
        norm = np.linalg.norm(delta, axis = 1)

        #Give an angle [-Pi, Pi] with arctan(theta) = y/x
        angles = np.arctan2(delta[:, 0], delta[:, 1]) 

        # If cell_size is available, convert norms to cm
        if hasattr(self, 'cell_size') and self.cell_size is not None:
            norm_cm = norm * self.cell_size
            vector = list(zip(norm_cm, angles))
        else:
            vector = list(zip(norm, angles))

        return vector 