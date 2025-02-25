from pitop import Pitop, Camera, PanTiltController, KeyboardButton
from pitop.processing.core.vision_functions import (
    center_reposition,
    find_centroid,
    get_object_target_lock_control_angle
)
from vendor.drive_controller import DriveController
from typing import Literal, Tuple
import time
import cv2
import numpy as np
import os
from collections import deque


class RemoteControls():
    FORWARD = "w"
    BACKWARD = "s"
    LEFT = "a"
    RIGHT = "d"
    CLAW_SERVO_UP = "e"
    CLAW_SERVO_DOWN = "r"
    CAMERA_SERVO_UP ="f"
    CAMERA_SERVO_DOWN = "g"
    QUIT = "esc"


class PitopRobot():
    def __init__(
        self,
        motor_ports: tuple[str, str] = ("M1", "M0"),
        claw_servo_port: str = "S2",
        camera_servo_port: str = "S1",
        navigation_method: Literal["remote", "autonomous"] = "autonomous",
        ordinance_template: str | None = None,
    ):
        self.pitop = Pitop()
        self.pitop.add_component(
            DriveController(
                left_motor_port=motor_ports[0],
                right_motor_port=motor_ports[1]
            )
        )
        self.ordinance_template = ordinance_template 
        self.navigation_method: Literal["remote", "autonomous"] = navigation_method
        self.claw_servo = PanTiltController(claw_servo_port)
        self.camera_servo = PanTiltController(camera_servo_port)
        self.drive_history: list[Tuple[str, int, int | None]] = []
        self.running = False
        self.distance_history = deque(maxlen=10)

        self.__run__()

    def pan_servo_to_angle(self, servo: PanTiltController, angle: int = 0):
        servo.pan_servo.target_speed = 100
        servo.pan_servo.target_angle = angle


    def tilt_servo_to_angle(self, servo: PanTiltController, angle: int = 0):
        servo.tilt_servo.target_speed = 100
        servo.tilt_servo.target_angle = angle

    
    def drive(
        self, 
        direction: Literal["forward", "backward", "left", "right", "stop"],
        distance: int | None,
        hold: bool = False,
        speed: int = 0.5,
        duration: int = 0
    ):
        self.drive_history.append((direction, hold, distance, duration))

        if direction == "forward":
            self.pitop.drive.forward(speed, hold, distance)
        elif direction == "backward":
            self.pitop.drive.backward(speed, hold, distance)
        elif direction == "left":
            self.pitop.drive.left(speed, hold, distance)
        elif direction == "right":
            self.pitop.drive.right(speed, hold, distance)
        elif direction == "stop":
            self.pitop.drive.stop()


    def reverse_drive_from_history(self, length: int):
        if len(self.drive_history) >= length:
            reversed_commands = self.drive_history[-length:]
            for command in reversed(reversed_commands):
                direction, hold, distance, duration = command
                if direction == "forward":
                    self.drive("backward", hold=hold, distance=distance)
                elif direction == "backward":
                    self.drive("forward", hold=hold, distance=distance)
                elif direction == "left":
                    self.drive("right", hold=hold, distance=distance)
                elif direction == "right":
                    self.drive("left", hold=hold, distance=distance)

                time.sleep(duration)

    def remote_navigation(self):
        target_claw_angle = 0
        target_camera_angle = 0

        self.pan_servo_to_angle(self.claw_servo, target_claw_angle)
        self.pan_servo_to_angle(self.camera_servo, target_camera_angle)

        def exit_loop():
            self.running = False

        
        def pan_claw(angle: int):
            target_claw_angle += angle
            self.pan_servo_to_angle(self.claw_servo, target_claw_angle)


        def pan_camera(angle: int):
            target_camera_angle += angle
            self.pan_servo_to_angle(self.camera_servo, target_camera_angle)

        control_map = {
            RemoteControls.FORWARD: lambda: self.drive("forward", hold=True, speed=1),
            RemoteControls.BACKWARD: lambda: self.drive("backward", hold=True, speed=1),
            RemoteControls.LEFT: lambda: self.drive("left", hold=False, speed=1),
            RemoteControls.RIGHT: lambda: self.drive("right", hold=False, speed=1),
            RemoteControls.CLAW_SERVO_UP: lambda: pan_claw(1),
            RemoteControls.CLAW_SERVO_DOWN: lambda: pan_claw(-1),
            RemoteControls.CAMERA_SERVO_UP: lambda: pan_camera(1),
            RemoteControls.CAMERA_SERVO_DOWN: lambda: pan_camera(-1),
            RemoteControls.QUIT: exit_loop,
        }

        buttons = {key: KeyboardButton(key) for key in control_map}

        pressed_keys = set()
        while self.running:
            for key, button in buttons.items():
                if button.is_pressed:
                    if key not in pressed_keys:
                        pressed_keys.add(key)
                        control_map[key]()
                else:
                    if key in pressed_keys:
                        pressed_keys.remove(key)
                        self.drive("stop")

            time.sleep(0.05)


    def autonomous_navigation(self):
        # Line HSV ranges
        lower_line_hsv = [120, 255, 255]
        upper_line_hsv = [100, 255, 255]

        # Ordinance HSV ranges
        lower_ordinance_hsv = [179, 255, 255]
        upper_ordinance_hsv = [179, 255, 127]

        # Frame size
        resolution = (640, 480)

        # Check for ordinance template
        resized_template_name = "resized_template.png"
        resized_template = None

        if os.path.isfile(resized_template_name.lower()):
            resized_template = cv2.imread(resized_template_name.lower())
        else:
            if os.path.isfile(self.ordinance_template) is not None:
                template = cv2.imread(self.ordinance_template)
                resized_template = cv2.resize(template, resolution)

                # Save the resized image if needed
                cv2.imwrite(resized_template_name.lower(), resized_template)

        # Add camera
        self.pitop.add_component(Camera(format="OpenCV"), rotate_angle=90, frame_resolution=resolution)

        def detect_line(frame):
            # Apply Gaussian blur to smooth the frame
            blurred_frame = cv2.GaussianBlur(frame, (9, 9), 0)

            # Convert BGR to HSV
            hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            # Threshold the HSV image to get only the target color
            mask = cv2.inRange(hsv_frame, lower_line_hsv, upper_line_hsv)

            # Apply Canny edge detection on the mask (not the original frame)
            edges = cv2.Canny(mask, 50, 150)

            # Find contours on the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                if largest_contour is not None:
                    centroid = find_centroid(largest_contour)
                    repositioned_centroid = center_reposition(centroid, frame)
                    angle = get_object_target_lock_control_angle(centroid, frame)
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    return repositioned_centroid, angle, w

            return None, None


        def get_hu_moments(contour):
            # Calculate moments for the contour
            moments = cv2.moments(contour)
            
            # Calculate Hu Moments
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Optional: Apply log transformation to make them more distinct
            for i in range(0, 7):
                if hu_moments[i] != 0:
                    hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))
            
            return hu_moments


        def compare_hu_moments(hu_moments1, hu_moments2):
            # Compare Hu Moments using a distance metric (e.g., Euclidean distance)
            distance = np.linalg.norm(hu_moments1 - hu_moments2)
            return distance


        def detect_ordinance(frame):
            # Apply Gaussian blur to smooth the frame
            blurred_frame = cv2.GaussianBlur(frame, (9, 9), 0)

            # Convert BGR to HSV
            hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            # Threshold the HSV image to get only the target color
            mask = cv2.inRange(hsv_frame, lower_ordinance_hsv, upper_ordinance_hsv)

            # Find contours on the thresholded mask
            contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by area (largest first)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Ensure the first contour is the image outline by ignoring it
            if len(sorted_contours) > 1:
                # Extract the second largest contour (actual shape)
                template_contour = sorted_contours[1]

                # Compute the Hu Moments for the template contour
                template_hu_moments = get_hu_moments(template_contour)

                closest_contour = None
                for c in sorted_contours[1:]:  # Skip the first contour (image outline)
                    # Compute Hu Moments for the current contour
                    hu_moments = get_hu_moments(c)

                    # Compare the Hu Moments between the template and current contour
                    distance = compare_hu_moments(template_hu_moments, hu_moments)
                    print(f"Hu Moments distance: {distance}")

                    # If distance is low (less than a threshold), consider it a valid match
                    if distance < 0.15:  # Threshold can be adjusted
                        closest_contour = c
                        
                        # Calculate centroid and angle of the matched contour
                        centroid = find_centroid(closest_contour)
                        repositioned_centroid = center_reposition(centroid, frame)
                        angle = get_object_target_lock_control_angle(centroid, frame)
                        x, y, w, h = cv2.boundingRect(closest_contour)

                        # Return the matched centroid and angle
                        return repositioned_centroid, angle, w
                else:
                    print("No valid match found.")
                    return None
            else:
                print("Not enough contours found.")
                return None
                

        def follow_angle(angle):
            # Proportional speed based on the angle, ensuring the max speed doesn't exceed 1
            drive_speed = max(0.1, min(abs(angle) / 360, 1))  # Scale the speed based on the angle (max speed = 1)
            self.drive("forward", speed=drive_speed, hold=True)  # Move forward with proportional speed
            self.pitop.drive.target_lock_drive_angle(angle)  # Lock target direction based on angle


        def estimate_distance_to_ordinance(ordinance_width, bbox_width):
            KNOWN_WIDTH = ordinance_width
            FOCAL_LENGTH = 420

            # Check if bbox_width is within a reasonable range to avoid errors
            if bbox_width < 10:  # Too small, return None or ignore
                print("Bounding box is too small, skipping distance calculation.")
                return None
            elif bbox_width > 500:  # Too large, return None or ignore
                print("Bounding box is too large, skipping distance calculation.")
                return None

            # Calculate the distance using the formula
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width

            # Define a valid distance range (e.g., between 20cm and 500cm)
            if distance < 15 or distance > 40:
                print(f"Calculated distance {distance} is out of range, skipping.")
                return None

            # Store the distance and calculate moving average
            self.distance_history.append(distance)
            smoothed_distance = sum(self.distance_history) / len(self.distance_history)

            return smoothed_distance
            
        try:
            # Initial Setup
            capturing_ordinance = False
            initial_claw_angle = -45
            initial_camera_angle = 12
            ordinance_width = 5.0
            target_distance = 19  # Target distance in cm (19 cm)
            alignment_threshold = 10  # Angle threshold (in degrees) to start aligning

            self.pan_servo_to_angle(self.claw_servo, initial_claw_angle)
            self.pan_servo_to_angle(self.camera_servo, initial_camera_angle)

            while self.running:
                frame = self.pitop.camera.get_frame()
                if frame is None:
                    continue

                # 1. Detect the line and ordinance
                line_centroid, line_angle, line_bbox_width = detect_line(frame)
                ordinance_centroid, ordinance_angle, ordinance_bbox_width = detect_ordinance(frame)
                
                # If no ordinance is detected, continue scanning
                if ordinance_centroid is None:
                    print("No ordinance detected, continuing scan.")
                else:
                    # Estimate the distance to the ordinance
                    ordinance_distance = estimate_distance_to_ordinance(ordinance_width, ordinance_bbox_width)

                    # If the distance is invalid, skip this loop
                    if ordinance_distance is None:
                        continue

                    print(f"Ordinance detected at distance: {ordinance_distance} cm.")

                    # 2. Adjust movement based on the distance to the ordinance
                    distance_error = ordinance_distance - target_distance

                    # 3. Adjust robot orientation based on the ordinance angle
                    if abs(ordinance_angle) > alignment_threshold:
                        print(f"Adjusting orientation to ordinance. Current angle: {ordinance_angle}")
                        self.drive("stop", distance=None, speed=0)  # Stop moving temporarily for adjustment
                        self.pitop.drive.target_lock_drive_angle(ordinance_angle)  # Rotate to align with ordinance
                        continue  # Skip further movement while aligning

                    # 4. Line-following logic: adjust robot heading based on the line angle
                    if line_centroid is not None and abs(line_angle) > 5:  # If line is detected and the angle is significant
                        print(f"Adjusting orientation to stay on the line. Line angle: {line_angle}")
                        self.drive("stop", distance=None, speed=0)  # Stop temporarily to adjust
                        self.pitop.drive.target_lock_drive_angle(line_angle)  # Adjust to stay aligned with the line

                    # 5. Adjust movement based on distance error to the ordinance
                    if distance_error > 0:
                        print(f"Too far from ordinance, moving forward. Distance error: {distance_error} cm.")
                        self.drive("forward", distance=None, speed=0.5)

                    elif distance_error < 0:
                        print(f"Too close to ordinance, stopping. Distance error: {distance_error} cm.")
                        self.drive("stop", distance=None, speed=0)  # Or move backward slightly if needed

                    else:
                        print("At the desired distance of 19 cm.")
                        self.drive("stop", distance=None, speed=0)  # Stay still

        except KeyboardInterrupt:
            self.running = False
            print("Stopping autonomous navigation.")


    def __run__(self):
        self.running = True
        if self.navigation_method == "remote":
            self.remote_navigation()
        elif self.navigation_method == "autonomous":
            self.autonomous_navigation()
        
