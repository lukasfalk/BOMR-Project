import numpy as np
import cv2
import glob

def calibrate_camera():
    # Taille du damier (nombre de coins INTERNES)
    CHECKERBOARD = (9, 6)  # Ajustez selon votre damier
    
    # Préparer les points 3D
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    objpoints = []  # Points 3D
    imgpoints = []  # Points 2D
    
    # Capturer 15-20 images du damier sous différents angles
    cap = cv2.VideoCapture(0)
    
    print("Appuyez sur ESPACE pour capturer, ESC pour terminer")
    images_captured = 0
    
    while images_captured < 20:
        ret, frame = cap.read()
        #print(frame.shape) 
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Trouver les coins du damier
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        frame_display = frame.copy()
        if ret:
            cv2.drawChessboardCorners(frame_display, CHECKERBOARD, corners, ret)
        
        cv2.putText(frame_display, f"Images: {images_captured}/20", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Calibration', frame_display)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32 and ret:  # SPACE
            objpoints.append(objp)
            imgpoints.append(corners)
            images_captured += 1
            print(f"Image {images_captured}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Sauvegarder les paramètres
    np.savez('camera_calibration.npz',
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
    
    print("\nCalibration done!")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs}")
    
    return camera_matrix, dist_coeffs


calibrate_camera()