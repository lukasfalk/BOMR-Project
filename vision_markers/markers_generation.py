# generate_markers.py - À exécuter UNE SEULE FOIS
import cv2

print("Le script markers_generation.py s'exécute correctement.")
print("OpenCV version:", cv2.__version__)

def generating_markers() :
    print("test")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Générer les 4 markers pour les coins de la map (5x5 cm à imprimer)
    for marker_id in range(4):
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
        cv2.imwrite(f'vision_markers/marker_corner_{marker_id}.png', marker)
        print(f"✓ Marker coin {marker_id} généré")

    # Générer le marker du robot (2-3 cm à imprimer)
    robot_marker = cv2.aruco.generateImageMarker(aruco_dict, 10, 200)
    cv2.imwrite('vision_markers/marker_robot_10.png', robot_marker)
    print("✓ Marker robot (ID 10) généré")

    print("\n📄 Imprimez les markers et collez-les sur votre map et robot!")

if __name__ == "__main__": #Calling function using in terminal : python vision_markers/markers_generation.py
    print("Test2")
    generating_markers()