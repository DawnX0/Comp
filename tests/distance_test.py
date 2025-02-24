from pitop import Pitop, Camera, PanTiltController
import cv2
import os
import time
import numpy as np
from collections import deque

# Keep the last N distance measurements
distance_history = deque(maxlen=10)

robot = Pitop()
robot.add_component(Camera(rotate_angle=90, format="OpenCV"))

ordinance_template = "ordinance_template.png"

# Servo ports
CAMERA_SERVO_PORT = "S1"
CLAW_SERVO_PORT = "S2"

claw_servo = PanTiltController(servo_pan_port=CLAW_SERVO_PORT)
camera_servo = PanTiltController(servo_pan_port=CAMERA_SERVO_PORT)

# ---------------------------
# Servo Control Functions
# ---------------------------
def pan_servo_to_angle(servo: PanTiltController, angle: int = 0):
    servo.pan_servo.target_angle = angle

def tilt_servo_to_angle(servo: PanTiltController, angle: int = 0):
    servo.tilt_servo.target_angle = angle

# ---------------------------
# Ordinance Detection Functions
# ---------------------------
def detect_ordinance_contour(frame):
    """
    Detects the edges of the ordinance in the provided image frame using HSV color thresholding
    to detect red color, with enhancements for robustness under varying lighting conditions.
    
    Parameters:
    frame (image): The input image frame to process for ordinance contour detection.

    Returns:
    image: The processed image with detected red areas highlighted.
    """
    
    # Step 1: Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Step 2: Normalize brightness with histogram equalization on the V channel
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v])
    
    # Step 3: Define the lower and upper bounds for red color in HSV
    lower_red1 = np.array([0, 80, 80])      # Lower range for red (hue from 0 to 10)
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 80])    # Upper range for red (hue from 160 to 180)
    upper_red2 = np.array([180, 255, 255])

    # Step 4: Create masks for both red ranges and combine them
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Step 5: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # blurred = cv2.bilateralFilter(mask, 9, 75, 75)

    # Step 6: Morphological closing to fill small gaps in the detected red areas
    kernel = np.ones((9, 9), np.uint8)
    red_areas = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # Step 7: Perform edge detection using Canny
    edges = cv2.Canny(red_areas, 30, 100)

    # Display processed result (edges of red areas)
    robot.miniscreen.display_image(edges)

    return edges  # Return edges of red areas after all transformations


def estimate_distance_to_ordinance(bounding_box_width):
    KNOWN_WIDTH = 5.0  # Adjust to your object width
    FOCAL_LENGTH = 420  # Calibrate for your camera setup

    # Calculate the distance
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bounding_box_width

    # Store the distance and calculate moving average
    distance_history.append(distance)
    smoothed_distance = sum(distance_history) / len(distance_history)

    print(f"Bounding Box Width: {bounding_box_width} px | Smoothed Distance: {smoothed_distance:.2f} cm")
    return smoothed_distance


def find_ordinance_contour(frame):
    edges = detect_ordinance_contour(frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if 1000 < area < 30000:  # Adjust size threshold
            x, y, w, h = cv2.boundingRect(contour)
            centroid_x, centroid_y = x + w // 2, y + h // 2
            distance = estimate_distance_to_ordinance(w)

            print(f"Ordinance detected at ({centroid_x}, {centroid_y}) - Distance: {distance:.2f} cm")
            return centroid_x, centroid_y, distance

    return None


try:
    
    pan_servo_to_angle(claw_servo, -30)
    pan_servo_to_angle(camera_servo, -8)
    
    while True:
        frame = robot.camera.get_frame()
        if frame is None:
            pass
        
        ordinance_results = find_ordinance_contour(frame)
        if ordinance_results:
            centroid_x, centroid_y, distance = ordinance_results
        else:
            print("No ordinance results", end="\r")
        
except KeyboardInterrupt:
    print("Stopped")