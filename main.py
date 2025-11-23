import numpy as np
import matplotlib.pyplot as plt 

#import vision #from vision import Vision
import global_nav #from global_nav import GlobalNavigation
# import motion_control
# import filtering
# import local_avoidance

def main():
    #v = vision.Vision()
    #v.vision_test(20)

    gnav = global_nav.GlobalNavigation()
    path, explored, operation_count = gnav.grid_search()
    gnav.display_grid_with_path(path)
    gnav.display_colored_grid()
    # if path:
    #     print("A* path length =", len(path)-1, "\n", path)
    #     print("length explored:", len(explored))
    #     print("A* visualization")
    # else:
    #     print("No path found with A*")



if __name__ == "__main__":
    main()