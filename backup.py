# Dependencies
import cv2
import numpy as np
from time import sleep
from collections import deque
from PitopRobot import Pitop, PanTiltController, Camera, UltrasonicSensor
from vendor.line_detect import process_frame_for_line
from vendor.drive_controller import DriveController

# Pitop Class
class Robot():
    """
    
    Pitop robot class
    
    """
    def __init__(
        self,
        motor_ports: tuple[str, str]= ("M1", "M0"),
        claw_servo_port: str = "S2",
        camera_servo_port: str = "S1",
    ):
        """
        Initializes a Robot instance with components for movement, camera control, and ordinance detection.

        Parameters:
        motor_ports (tuple): A tuple specifying the ports for the left and right motors, default is ("M1", "M0").
        claw_servo_port (str): The port for controlling the claw servo, default is "S2".
        camera_servo_port (str): The port for controlling the camera servo, default is "S1".
        ordinance_template_path (str): Path to the image file used for ordinance template matching, default is "ordinance_template.png".

        Components added to the robot:
        - DriveController: Controls the robot's left and right motors, using the specified motor ports.
        - Camera: A camera component is added, configured to use the OpenCV format and rotated by 90 degrees.
        - PanTiltController (for claw and camera): These controllers manage the movement of the claw and camera servos.

        The Pitop instance is used to manage the robot's components.
        An assertion ensures that the provided ordinance template path is valid.
        """
        self.pitop = Pitop()
        self.pitop.add_component(
            DriveController(
                left_motor_port=motor_ports[0],
                right_motor_port=motor_ports[1]
            )
        )
        self.pitop.add_component(Camera(format="OpenCV", rotate_angle=90))
        self.claw_servo = PanTiltController(claw_servo_port)
        self.camera_servo = PanTiltController(camera_servo_port)
        self.claw_servo.pan_servo.target_speed = 100
        self.camera_servo.pan_servo.target_speed = 100
        self.distance_history = deque(maxlen=10)
        self.alignment_threshold = 0.02
        self.initial_claw_angle = -45
        self.initial_camera_angle = 12
        self.capturing_ordinance = False
    
    def pan_servo_to_angle(self, servo: PanTiltController, angle: int = 0):
        """
        Pans the servo to a specified angle.

        Parameters:
        servo (PanTiltController): The PanTiltController instance controlling the pan and tilt servos.
        angle (int): The target angle to pan the servo to, default is 0.

        This method adjusts the pan servo's target angle to the specified value.
        """
        servo.pan_servo.target_speed = 100
        servo.pan_servo.target_angle = angle

    def tilt_servo_to_angle(self, servo: PanTiltController, angle: int = 0):
        """
        Tilts the servo to a specified angle.

        Parameters:
        servo (PanTiltController): The PanTiltController instance controlling the pan and tilt servos.
        angle (int): The target angle to tilt the servo to, default is 0.

        This method adjusts the tilt servo's target angle to the specified value.
        """
        servo.tilt_servo.target_angle = angle


    def read_ultrasonic(self, ultrasonic: UltrasonicSensor):
        """
        Reads the distance measurement from the provided ultrasonic sensor.

        Parameters:
        ultrasonic (UltrasonicSensor): The ultrasonic sensor instance to read distance from.

        Returns:
        float: The distance measured by the ultrasonic sensor in meters.
        """
        return ultrasonic.distance
    
    
    def detect_line(self, frame):
        """
        Detects a line in the given image frame and returns the detected angle.

        Parameters:
        frame (image): The input image frame to process for line detection.

        Returns:
        int or None: The angle of the detected line, or None if no line is found or the frame is invalid.

        This function processes the provided frame to detect a line. If no line is detected, the robot's movement 
        is stopped and None is returned. If a line is detected, the robot's camera view is displayed, and the 
        calculated angle of the line is returned.
        """
        if frame is None:
            print("Error: No valid image frame!")
            return None

        processed_frame = process_frame_for_line(frame)

        if processed_frame.line_center is None:
            self.pitop.drive.stop()
            return None
        else:
            self.pitop.miniscreen.display_image(processed_frame.robot_view)
            return processed_frame.angle


    def detect_ordinance_contour(self, frame):
        """
        Detects the edges of the ordinance in the provided image frame using various transformations:
        grayscale, adaptive threshold, Gaussian blur, Canny edge detection, morphological transformations,
        and HSV thresholding to detect red color.

        Parameters:
        frame (image): The input image frame to process for ordinance contour detection.

        Returns:
        image: The processed image with detected edges, ready for contour finding.
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

        # Step 5: Apply Bilateral Blur to reduce noise and preserve edges
        bilateral_blur = cv2.bilateralFilter(mask, 9, 75, 75)
        
        # Step 6: Apply Median Blur to reduce salt and pepper noise
        median_blur = cv2.medianBlur(bilateral_blur, 5)
        
        # Step 6: Morphological closing to fill small gaps in the detected red areas
        kernel = np.ones((9, 9), np.uint8)
        red_areas = cv2.morphologyEx(median_blur, cv2.MORPH_CLOSE, kernel)

        # Step 7: Perform edge detection using Canny
        edges = cv2.Canny(red_areas, 70, 200)

        # Display processed result (edges of red areas)
        # self.pitop.miniscreen.display_image(edges)

        return edges  # Return edges of red areas after all transformations


    def estimate_distance_to_ordinance(self, bounding_box_width):
        """
        Estimates the distance from the camera to the ordinance based on the bounding box width.

        Parameters:
        bounding_box_width (int): The width of the bounding box around the ordinance in pixels.

        Returns:
        float: The estimated distance to the ordinance in centimeters.

        This method uses a simple camera calibration formula to estimate the distance, based 
        on a known object width (KNOWN_WIDTH) and the camera's focal length (FOCAL_LENGTH).
        """
        KNOWN_WIDTH = 5.0  # Adjust to your object width
        FOCAL_LENGTH = 420  # Calibrate for your camera setup

        # Calculate the distance
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bounding_box_width

        # Store the distance and calculate moving average
        self.distance_history.append(distance)
        smoothed_distance = sum(self.distance_history) / len(self.distance_history)

        # print(f"Bounding Box Width: {bounding_box_width} px | Smoothed Distance: {smoothed_distance:.2f} cm")
        return smoothed_distance


    def find_ordinance_contour(self, frame):
        """
        Finds and returns the contour of the ordinance in the image frame along with its distance.

        Parameters:
        frame (image): The input image frame to find the ordinance contour.

        Returns:
        tuple or None: A tuple (centroid_x, centroid_y, distance) if ordinance is detected, 
        or None if no ordinance is found.
        """
        edges = self.detect_ordinance_contour(frame)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if 1500 < area < 25000:  # Adjust size threshold
                x, y, w, h = cv2.boundingRect(contour)
                centroid_x, centroid_y = x + w // 2, y + h // 2
                distance = self.estimate_distance_to_ordinance(w)

                print(f"Ordinance detected at ({centroid_x}, {centroid_y}) - Distance: {distance:.2f} cm", end="\r")
                return centroid_x, centroid_y, distance

        return None
    
    def pickup_ordinance_sequence(self, relative_centroid_x):
        # Check if lined up
        if abs(relative_centroid_x) < self.alignment_threshold:
            self.capturing_ordinance = True   
            self.pitop.drive.stop()
            self.pan_servo_to_angle(self.claw_servo, 30)
            sleep(1)
            self.pitop.drive.forward(0.1, hold=True)
            sleep(3)
            self.pitop.drive.stop()
            self.pan_servo_to_angle(self.claw_servo, self.initial_claw_angle)
            sleep(1)
            self.capturing_ordinance = False
        else:
            if relative_centroid_x < 0:
                print("Aligned too far right, adjusting left.", end="\r")
                # Move left (adjust speed or turn left slightly)
                self.pitop.drive.stop()
                self.pitop.drive.left(0.035)
            elif relative_centroid_x > 0:
                print("Aligned too far left, adjusting right.", end="\r")
                # Move right (adjust speed or turn right slightly)
                self.pitop.drive.stop()
                self.pitop.drive.right(0.035)
    
    def run(self):
        drive_speed = 0.5

        self.pan_servo_to_angle(self.claw_servo, self.initial_claw_angle)
        self.pan_servo_to_angle(self.camera_servo, self.initial_camera_angle)
        
        try:
            while True:
                frame = self.pitop.camera.get_frame()
                if frame is None:
                    print("Skipping invalid frame...")
                    continue

                # 1. Detect ordinance
                ordinance_result = self.find_ordinance_contour(frame)

                # 2. Detect lines
                line_result = self.detect_line(frame)

                # 3. Movement Logic
                if ordinance_result and not self.capturing_ordinance:
                    centroid_x, centroid_y, distance = ordinance_result
                    
                    relative_centroid_x = (centroid_x - (frame.shape[1] / 2)) / (frame.shape[1] / 2)
                    
                    if 40 > distance > 25:
                        drive_speed = max(0.1, drive_speed - abs(relative_centroid_x) / 360)
                        self.pitop.drive.forward(drive_speed, hold=True)
                        self.pitop.drive.target_lock_drive_angle(relative_centroid_x)
                        print(f"Approaching ordinance - Current Distance: {distance}", end="\r")
                    elif 23 > distance > 19:
                        self.pickup_ordinance_sequence(relative_centroid_x)
                    else:
                        self.pitop.drive.stop()
                        self.pitop.drive.backward(0.1, hold=True)
                        
                elif line_result is not None and not self.capturing_ordinance:
                    drive_speed = max(0.25, drive_speed - abs(line_result) / 360)
                    self.pitop.drive.forward(drive_speed, hold=True)
                    self.pitop.drive.target_lock_drive_angle(line_result)
                    print("Following line", end="\r", flush=True)
                    
                else:
                    print("No ordinance or line detected. Stopping robot.", end="\r")
                    self.pitop.drive.stop()

        except KeyboardInterrupt:
            print("Stopping robot")
            self.pitop.drive.stop()


new_robot = Robot()
new_robot.run()