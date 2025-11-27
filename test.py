import numpy as np

thymio_size = 5  # Taille maximale du robot (en unités réelles)
cell_size = 1  # Taille d'une cellule (en unités réelles)

# Calculer la taille totale en cellules
robot_size_cells = int(np.ceil(thymio_size / cell_size)) #np.ceil to take he superior int
robot_radius = (robot_size_cells // 2) 
robot_size_cells = 2 * robot_radius + 1  # Rayon à gauche + centre + rayon à droite

print("Rayon du robot en cellules :", robot_radius)
print("Taille totale du robot en cellules :", robot_size_cells)