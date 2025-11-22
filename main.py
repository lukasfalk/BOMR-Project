import numpy as np
import matplotlib.pyplot as plt 

import vision #from vision import Vision
import global_nav #from global_nav import GlobalNavigation
import motion_control
import filtering
import local_avoidance

def main():
    vision = vision.Vision()
    nav = global_nav.GlobalNavigation(vision)


if __name__ == "__main__":
    main()