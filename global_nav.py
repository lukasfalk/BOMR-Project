import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from heapq import heappush, heappop


class GlobalNavigation : 

    def __init__(self):
       
        
        # self.vision = vision #To get grid, thymio_pos, thymio_size, cell_size and goal
        # self.Start = self.vision.thymio_pos
        # self.Goal = self.vision.goal
        # self.grid = self.vision.grid
        # self.cell_size = self.vision.cell_size
        # self.robot_size = self.vision.thymio_size 
        # self.robot_c_width = int(self.vision.thymio_size[0] / self.cell_size[0])
        # self.robot_c_height = int(self.vision.thymio_size[1] / self.cell_size[1])
        '''
        Testing zone 
        '''
        # Testing zone: Generate a large grid for testing
        self.grid = np.ones((10, 10), dtype=int)  # Create a 20x20 grid filled with zeros
        self.Start = (0, 0)  # Top-left corner
        self.Goal = (9, 9)  # Bottom-right corner

        self.thymio_size = 1.5  # Taille maximale du robot (en unités réelles)
        self.cell_size = 1  # Taille d'une cellule (en unités réelles)

        # Calculer la taille totale en cellules
        self.robot_size = int(np.ceil(self.thymio_size / self.cell_size)) // 2 #np.ceil to take he superior int
        self.robot_size_cells = 2 * self.robot_size + 1  # Rayon à gauche + centre + rayon à droite

        print("Rayon du robot en cellules :", self.robot_size)
        print("Taille totale du robot en cellules :", self.robot_size_cells)

        # Add some obstacles to the grid for testing
        self.grid[5, 5] = 0  # Small block
        # self.grid[10, 5:15] = 0  # Horizontal wall
        # self.grid[15:18, 10:12] = 0  # Small block

        # Print the grid for visualization
        print("Generated grid for testing:")
        print(self.grid)

        '''
        End Testing zone
        '''
    
    #Function to verify if the robot is inside the grid and not on an obstacle.
    def verification(self, position):
        
        x, y = position #Top left position
        
    
        for i in range(x, x + self.robot_c_height):
            for j in range(y, y + self.robot_c_width):
                
                if i >= len(self.grid) or j >= len(self.grid[0]):
                    print("Le robot dépasse les limites de la grille.")
                    return False
                elif self.grid[i][j] == 0 :
                    print("Erreur : Le robot touche un mur.")
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
                            
                            print("Distance di selon x : ", di)
                            ni = i + di
                            nj = j + dj
                            if 0 <= ni < len(self.grid) and 0 <= nj < len(self.grid[0]):
                                modified_grid[ni][nj] = 0 # adding obstacles

        self.grid = modified_grid

    def heuristic(self, a, b):
        # Implement the Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    #Path finding algorithme with A* and Manhattan distance 
    def grid_search(self): 
        
        #Function to adapt the path for thymio robot dimension
        self.growing_obstacles()
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

            # Get neighbors (up, down, left, right) -> be careful it is like a matrix
            neighbors = [
                (current_pos[0]-1, current_pos[1]),  # Up
                (current_pos[0]+1, current_pos[1]),  # Down
                (current_pos[0], current_pos[1]-1),  # Left
                (current_pos[0], current_pos[1]+1)   # Right
            ]

            for neighbor in neighbors: # for each neighbor of marked cells
                # Check if neighbor is within bounds
                if (0 <= neighbor[0] < len(self.grid)) and (0 <= neighbor[1] < self.grid.shape[1]): #check if the neighbor is less than 1 and greater than 0
                    # Check if neighbor is not an obstacle

                    if self.grid[neighbor[0], neighbor[1]] != 0:

                        if neighbor in explored:
                            continue

                        tentative_g_cost = current_g_cost + self.grid[neighbor[0], neighbor[1]]

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

        # Marquer le chemin sur la grille
        for row, col in path:
            grid_with_path[row, col] = 2  # Utiliser '2' pour représenter le chemin
        
        #Assign start and goal for different color on map
        start_y, start_x = path[0]
        grid_with_path[start_y, start_x] = 3

        goal_y, goal_x = path[-1]
        grid_with_path[goal_y, goal_x] = 4

        self.grid = grid_with_path
        # Afficher la grille avec le chemin
        print("Grid with path:")
        print(grid_with_path)

   
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