import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop


class GlobalNavigation : 

    def __init__(self, vision):
       
        self.vision = vision #To get grid, thymio_pos, thymio_size, cell_size and goal
        self.Start = self.vision.thymio_pos
        self.Goal = self.vision.goal
        self.grid = self.vision.grid
        self.cell_size = self.vision.cell_size
        self.robot_size = self.vision.thymio_size 

    def heuristic(a, b):
        # Implement the Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def robot_verification(self):
        
        x_start, y_start = self.Start
        cell_width, cell_height = self.cell_size
        robot_width, robot_height = self.robot_size

        # Convertir la taille du robot en nombre de cellules
        robot_cells_width = int(robot_width / cell_width)
        robot_cells_height = int(robot_height / cell_height)

        # Parcourir la zone occupée par le robot
        for i in range(x_start, x_start + robot_cells_height):
            for j in range(y_start, y_start + robot_cells_width):
                # Vérifier si on sort de la grille
                if i >= len(grid) or j >= len(grid[0]):
                    raise ValueError("Le robot dépasse les limites de la grille.")
                # Vérifier si la cellule est un mur
                if grid[i][j] == 1:
                    raise ValueError("Le robot touche un mur à la position physique ({}, {}).".format(
                        i * cell_height, j * cell_width
                    ))

        if condition_erreur:
            print("Erreur : Le robot touche un mur.")
            return False
        return True

    def grid_search(self):
        

        #Redifine robot_size as a radius from his center so we have a circle to not touch the walls
        robot_size = max(robot_size)/2 

        ## initialize the varibales above
        came_from = {}      # to reconstruct path
        g_costs = {S: 0}    # cost from start to the cell
        explored = set()    # to keep track of explored cells
        operation_count = 0 # to count the number of operations


        open_set = [(self.heuristic(Start, Goal), 0, Start)]  # priority queue for A* (f_cost, g_cost, position)
            
        while open_set: # Goal is unmarked
        
            current_f_cost, current_g_cost, current_pos = heappop(open_set)

            explored.add(current_pos)
            
            if current_pos == Goal: # if G goal is marked
                break

            # Get neighbors (up, down, left, right)
            neighbors = [
                (current_pos[0]-1, current_pos[1]),  # Up
                (current_pos[0]+1, current_pos[1]),  # Down
                (current_pos[0], current_pos[1]-1),  # Left
                (current_pos[0], current_pos[1]+1)   # Right
            ]

            for neighbor in neighbors: # for each neighbor of marked cells
                # Check if neighbor is within bounds
                if (0 <= neighbor[0] < map_grid.shape[0]) and (0 <= neighbor[1] < map_grid.shape[1]): #check if the neighbor is less than 1 and greater than 0
                    # Check if neighbor is not an obstacle
                    if map_grid[neighbor[0], neighbor[1]] != -1:
                        tentative_g_cost = current_g_cost + 1  + map_grid[neighbor[0], neighbor[1]]

                        if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                            g_costs[neighbor] = tentative_g_cost
                            came_from[neighbor] = current_pos
                            operation_count += 1 
                        
                            f_cost = tentative_g_cost + self.heuristic(neighbor, Goal)
                            heappush(open_set, (f_cost, tentative_g_cost, neighbor))
                            
        # Reconstruct path
        if current_pos == Goal:
            path = []
            while current_pos != Start:
                path.append(current_pos)
                current_pos = came_from[current_pos]
            path.append(Start)
            path.reverse()
            return path, explored, operation_count  # Return reversed path and explored cells
        else:
        # If we reach here, no path was found
            return None, explored, operation_count
        

    path, explored, operation_count = grid_search(Map, Start, Goal, algo="A*")
    
    if path:
        print("A* path length =", len(path)-1, "\n", path)
        print("length explored:", len(explored))
        print("A* visualization")
        display_map(Map, path, Start, Goal, explored)
    else:
        print("No path found with A*")



#Need map (grid with grid_Nx and grid_Ny and cell size)