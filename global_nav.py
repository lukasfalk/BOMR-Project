import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from heapq import heappush, heappop


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

        # '''
        # Testing zone 
        # '''
        # # Testing zone: Generate a large grid for testing
        # self.grid = np.ones((10, 10), dtype=int)  # Create a 20x20 grid filled with zeros
        # self.Start = (0, 0)  # Top-left corner
        # self.Goal = (9, 9)  # Bottom-right corner

        # self.thymio_size = 1.5  # Taille maximale du robot (en unités réelles)
        # self.cell_size = 1  # Taille d'une cellule (en unités réelles)

        # # Calculer la taille totale en cellules
        # self.robot_size = int(np.ceil(self.thymio_size / self.cell_size)) // 2 #np.ceil to take he superior int
        # self.robot_size_cells = 2 * self.robot_size + 1  # Rayon à gauche + centre + rayon à droite

        # print("Rayon du robot en cellules :", self.robot_size)
        # print("Taille totale du robot en cellules :", self.robot_size_cells)

        # # Add some obstacles to the grid for testing
        # self.grid[5, 5] = 0  # Small block
        # # self.grid[10, 5:15] = 0  # Horizontal wall
        # # self.grid[15:18, 10:12] = 0  # Small block

        # # Print the grid for visualization
        # print("Generated grid for testing:")
        # print(self.grid)

        # '''
        # End Testing zone
        # '''

    def set_gnav(self, vision):

        row, col = vision.thymio_pos
        self.Start = row, col
        row, col = vision.goal
        self.Goal = col, row
        self.grid = vision.grid
        self.cell_size = vision.cell_size
        self.thymio_size = 12

        #Adapt thymio size for grid
        self.robot_size = int(np.ceil(self.thymio_size / self.cell_size)) // 2 #np.ceil to take he superior int
        self.robot_size_cells = 2 * self.robot_size + 1  # Rayon à gauche + centre + rayon à droite

        '''
        Testing
        '''
        print("Rayon du robot en cellules :", self.robot_size)
        print("Taille totale du robot en cellules :", self.robot_size_cells)
        '''
        Stop Testing
        '''

    
    
    #Function to verify if the robot is inside the grid and not on an obstacle.
    def verification(self):
        col, row = self.Start
        print("Ce qu'on check à droite : ", -self.robot_size + col)
        print("Ce qu'on check à gauche : ", self.robot_size + col)
        for i in range(-self.robot_size + row, row + self.robot_size):
            for j in range(-self.robot_size + col, col + self.robot_size):
                
                if i >= len(self.grid) or j >= len(self.grid[0]):
                    print("Error : Thymio is not inside the grid !")
                    return False
                elif self.grid[i][j] == 0 :
                    print("Error : Thymio is on an obstable !")
                    return False

        return True
    
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

    # def heuristic(self, a, b):
    #     # Implement the Manhattan distance heuristic
    #     return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
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
        self.growing_obstacles()
        a = self.verification()
        ## initialize the variables above
        came_from = {}      # to reconstruct path
        g_costs = {self.Start: 0}    # cost from start to the cell
        explored = set()    # to keep track of explored cells
        operation_count = 0 # to count the number of operations

        open_set = [(self.heuristic(self.Start, self.Goal), 0, self.Start)]  # priority queue for A* (f_cost, g_cost, position)
            
        while open_set: # Goal is unmarked
        
            current_f_cost, current_g_cost, current_pos = heappop(open_set) #Take the last position in grid

            explored.add(current_pos) 
            
            if current_pos == self.Goal: # if Goal is marked then stop the algorithme
                break

            # Get neighbors -> be careful it is like a matrix
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
                if (0 <= neighbor[0] < len(self.grid)) and (0 <= neighbor[1] < self.grid.shape[1]): #check if the neighbor is less than 1 and greater than 0
                    
                    # Check if neighbor is not an obstacle
                    if self.grid[neighbor[0], neighbor[1]] != 0 :

                        if neighbor in explored:
                            continue

                        if (abs(neighbor[0] - current_pos[0]) == 1 and abs(neighbor[1] - current_pos[1]) == 1) :
                            is_diagonal = True 
                        else : 
                            is_diagonal = False 

                        if is_diagonal : 
                            move_cost = np.sqrt(2) #Diagonal cost is sqrt(2)
                        else : 
                            move_cost = 1 #Orthogonal cost is 1 
                        
                        tentative_g_cost = current_g_cost + move_cost
                        #tentative_g_cost = current_g_cost + self.grid[neighbor[0], neighbor[1]]

                        if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                            g_costs[neighbor] = tentative_g_cost
                            came_from[neighbor] = current_pos
                            operation_count += 1 
                        
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
            return path, explored, operation_count  # Return reversed path and explored cells
        else:
        # If we reach here, no path was found
            return None, explored, operation_count
        
    def display_grid_with_path(self, path):

        # Créer une copie de la grille pour ne pas modifier l'originale
        grid_with_path = self.grid.copy()

        thickness = 5  # Épaisseur pour le chemin, start et goal
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
        # Afficher la grille avec le chemin
        #print("Grid with path:")
        #print(grid_with_path)

   
    def display_colored_grid(self):

        # 0 = black, 1 = white, 2 = yellow, 3 = red, 4 = blue"
        cmap = ListedColormap(['black', 'white', 'yellow', 'red', 'blue']) 

    
        plt.imshow(self.grid, cmap=cmap, origin='upper')

        # Add lines between cells
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, self.grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

        plt.title("Colored Grid Visualization")
        plt.xlabel("X-axis (Columns)")
        plt.ylabel("Y-axis (Rows)")
        plt.show()