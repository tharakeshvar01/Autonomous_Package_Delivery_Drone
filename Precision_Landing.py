import cv2
import numpy as np
from pymavlink import mavutil

# Load camera calibration data
calibration_data = np.load('calibration_data.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Initialize the ArUco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Initialize the ArUco parameters
parameters = cv2.aruco.DetectorParameters_create()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize drone connection
mav = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

# Target ArUco marker ID and side length in meters
target_marker_id = 0
marker_side_length = 0.1  # meters

# Function to control the drone to land on the marker
def land_on_marker(marker_pose):
    # Define desired altitude above the marker
    target_altitude = 1.0  # meters
    
    # Extract x, y, and z coordinates of the marker
    x, y, z = marker_pose[:, 3]

    # Send MAVLink messages to control the drone's position and altitude
    msg = mavutil.mavlink.MAVLink_position_target_local_ned_message(
        time_boot_ms=0,
        target_system=1,
        target_component=1,
        coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask=mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_Y,
        x=x,
        y=y,
        z=target_altitude,
        vx=0,
        vy=0,
        vz=0,
        afx=0,
        afy=0,
        afz=0,
        yaw=0,
        yaw_rate=0
    )
    mav.send(msg)

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and target_marker_id in ids:
        # Get index of the target marker
        marker_index = np.where(ids == target_marker_id)[0][0]

        # Estimate the pose of the target marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index], marker_side_length, camera_matrix, dist_coeffs)

        # Draw axis on the marker
        cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs, tvecs, marker_side_length)

        # Control the drone to land on the marker
        land_on_marker(tvecs)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
