import cv2
import time
import numpy as np
from PitopRobot import Pitop, UltrasonicSensor, Camera, PanTiltController
from vendor.drive_controller import DriveController
from vendor.line_detect import process_frame_for_line

# Motor ports
LEFT_MOTOR_PORT = "M1"
RIGHT_MOTOR_PORT = "M0"

# Ultrasonic ports
FRONT_ULTRASONIC_PORT = "A1"

# Servo ports
CAMERA_SERVO_PORT = "S1"
CLAW_SERVO_PORT = "S2"

# Constants
ULTRASONIC_THRESHOLD = 0.1       # For obstacle avoidance
ORDINANCE_DISTANCE = 0.4           # Distance threshold to stop at ordinance
TEMPLATE_CONFIDENCE = 0.7          # Confidence threshold to detect ordinance

# Template image for ordinance detection
ORDINANCE_TEMPLATE_PATH = "ordinance_template.png"

# Variables
drive_speed = 0.5  # Base drive speed
capturing_ordinance = False
front_ultrasonic = UltrasonicSensor(FRONT_ULTRASONIC_PORT)
claw_servo = PanTiltController(servo_pan_port=CLAW_SERVO_PORT)
camera_servo = PanTiltController(servo_pan_port=CAMERA_SERVO_PORT)

# Robot Setup
robot = Pitop()
robot.add_component(DriveController(
    left_motor_port=LEFT_MOTOR_PORT,
    right_motor_port=RIGHT_MOTOR_PORT
))
robot.add_component(Camera(format="OpenCV", rotate_angle=90))

# ---------------------------
# Servo Control Functions
# ---------------------------
def pan_servo_to_angle(servo: PanTiltController, angle: int = 0):
    servo.pan_servo.target_angle = angle

def tilt_servo_to_angle(servo: PanTiltController, angle: int = 0):
    servo.tilt_servo.target_angle = angle

# ---------------------------
# Line Detection Function
# ---------------------------
def detect_line(frame):
    if frame is None:
        print("Error: No valid image frame!")
        return None

    processed_frame = process_frame_for_line(frame)

    if processed_frame.line_center is None:
        robot.drive.stop()
        return None
    else:
        robot.miniscreen.display_image(processed_frame.robot_view)
        return processed_frame.angle

# ---------------------------
# Obstacle Detection Function
# ---------------------------
def detect_obstacle():
    """ Detects if an obstacle is too close. """
    return front_ultrasonic.distance <= ULTRASONIC_THRESHOLD

# ---------------------------
# Ordinance Detection Functions
# ---------------------------
def detect_ordinance_contour(frame):
    if frame is None:
        print("Error: No valid image frame!")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)  # Adjust thresholds as needed
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges


def estimate_distance_to_ordinance(bounding_box_width):
    # Adjust KNOWN_WIDTH based on the real-world width of the ordinance
    KNOWN_WIDTH = 7.0  # Example: 10 cm (adjust to your object)
    FOCAL_LENGTH = 720  # Calibrate this value for your camera setup

    # Distance calculation
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bounding_box_width
    return distance


def find_ordinance_contour(frame):
    edges = detect_ordinance_contour(frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1500:  # Adjust size threshold
            x, y, w, h = cv2.boundingRect(contour)
            centroid_x, centroid_y = x + w // 2, y + h // 2
            distance = estimate_distance_to_ordinance(w)
            
            print(f"Ordinance detected at ({centroid_x}, {centroid_y}) - Distance: {distance:.2f} cm")
            return centroid_x, centroid_y, distance  # Return distance too

    return None

# ---------------------------
# Movement Towards Ordinance
# ---------------------------
def move_towards_centroid(centroid_x, distance, frame):
    global capturing_ordinance

    robot.drive.stop()
    capturing_ordinance = True
    pan_servo_to_angle(claw_servo, 40)
    time.sleep(1)

    Kp_turn = 0.5  # Adjust as needed
    frame_center_x = frame.shape[1] // 2

    while True:
        normalized_offset = (centroid_x - frame_center_x) / frame_center_x
        turn_angle = normalized_offset * Kp_turn
        turn_angle = max(min(turn_angle, 1), -1)
        
        drift_correction = -0.34
        turn_angle += drift_correction

        # Adjust speed based on distance
        if distance > 50:  # Far away
            speed = 0.4
        elif distance > 30:  # Medium distance
            speed = 0.25
        else:  # Close
            time.sleep(1)
            break
        
        print(f"Approaching ordinance at {turn_angle:.2f} with speed {speed:.2f}", end="\r")

        robot.drive.forward(0.1, hold=True)
        robot.drive.target_lock_drive_angle(turn_angle)

        # Stop if close enough
        if front_ultrasonic.distance <= 0.3:
            break

    robot.drive.stop()
    pan_servo_to_angle(claw_servo, -50)
    time.sleep(2)
    
    ordinance_result = find_ordinance_contour(robot.camera.get_frame())
    centroid_x, centroid_y, distance = ordinance_result
    
    if distance:
        if distance < 20:
            robot.drive.backward(0.1)
            time.sleep(3)
            
        robot.drive.stop()
        pan_servo_to_angle(claw_servo, 50)
        time.sleep(2)
        robot.drive.forward(0.1, hold=True)
        time.sleep(3)
        pan_servo_to_angle(claw_servo, -50)
        time.sleep(2)

    capturing_ordinance = False

# ---------------------------
# Search for ordinance or line
# ---------------------------
def search():
    robot.drive.left(0.2)
    time.sleep(0.1)
    robot.drive.stop()
    time.sleep(1)

# ---------------------------
# Main Run Loop
# ---------------------------
def run():
    
    global drive_speed
    global capturing_ordinance

    pan_servo_to_angle(claw_servo, -30)
    pan_servo_to_angle(camera_servo, -8)
    
    try:
        while True:
            frame = robot.camera.get_frame()
            if frame is None:
                print("Skipping invalid frame...")
                continue

            # 1. Detect ordinance
            if not capturing_ordinance:  # Only check if we're not already moving toward it
                ordinance_result = find_ordinance_contour(frame)

            # 2. Detect lines
            line_result = detect_line(frame)

            # 3. Movement Logic
            if ordinance_result and not capturing_ordinance:
                centroid_x, centroid_y, distance = ordinance_result
                move_towards_centroid(centroid_x, distance, frame)
                time.sleep(2)

            elif line_result is not None and not capturing_ordinance:
                target_lock_drive_angle = line_result
                print(f"Following line at angle {target_lock_drive_angle:.2f}°", end="\r")
                drive_speed = max(0.1, drive_speed - abs(target_lock_drive_angle) / 360)
                robot.drive.forward(drive_speed, hold=True)
                robot.drive.target_lock_drive_angle(target_lock_drive_angle)
            else:
                print("No ordinance or line detected. Stopping robot.", end="\r")
                robot.drive.stop()

    except KeyboardInterrupt:
        print("Stopping robot..")
        pan_servo_to_angle(camera_servo, -5)
        robot.drive.stop()

# ---------------------------
# Start robot
# ---------------------------
run()