from pitop import Pitop, Camera, PanTiltController, KeyboardButton
from pitop.processing.core.vision_functions import (
    center_reposition,
    find_centroid,
    get_object_target_lock_control_angle
)
from vendor.drive_controller import DriveController
from vendor.line_detect import process_frame_for_line
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
        ordinance_template: str | None = "ordinance_template.png",
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

    
    def drive(self, direction: Literal["forward", "backward", "left", "right", "stop"],
            distance: int | None = None,
            hold: bool = False,
            speed: int = 0.5,
            duration: int = 0):
        
        if direction not in {"forward", "backward", "left", "right", "stop"}:
            raise ValueError(f"Invalid drive direction: {direction}")

        self.drive_history.append((direction, distance, hold, duration))

        drive_command = getattr(self.pitop.drive, direction, None)
        if drive_command and direction != "stop":
            drive_command(speed, hold, distance)
        else:
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
            self.pan_servo_to_angle(self.claw_servo, self.claw_servo.pan_servo.current_angle + angle)


        def pan_camera(angle: int):
            self.pan_servo_to_angle(self.camera_servo, self.camera_servo.pan_servo.current_angle + angle)

        control_map = {
            RemoteControls.FORWARD: lambda: self.drive("forward", hold=True, speed=1),
            RemoteControls.BACKWARD: lambda: self.drive("backward", hold=True, speed=1),
            RemoteControls.LEFT: lambda: self.drive("left", hold=False, speed=1),
            RemoteControls.RIGHT: lambda: self.drive("right", hold=False, speed=1),
            RemoteControls.CLAW_SERVO_UP: lambda: pan_claw(3),
            RemoteControls.CLAW_SERVO_DOWN: lambda: pan_claw(-3),
            RemoteControls.CAMERA_SERVO_UP: lambda: pan_camera(3),
            RemoteControls.CAMERA_SERVO_DOWN: lambda: pan_camera(-3),
            RemoteControls.QUIT: exit_loop,
        }

        buttons = {key: KeyboardButton(key) for key in control_map}

        pressed_keys = set()
        while self.running:
            for key, button in buttons.items():
                if button.is_pressed:
                    if key not in pressed_keys:
                        if key not in("e", "f", "r", "g"):
                            pressed_keys.add(key)
                        control_map[key]()
                else:
                    if key in pressed_keys:
                        pressed_keys.remove(key)
                        self.drive("stop")

            time.sleep(0.05)


    def autonomous_navigation(self):
        # Ordinance HSV ranges (Hot pink)
        lower_ordinance_hsv = np.array([140, 100, 100])
        upper_ordinance_hsv = np.array([170, 255, 255])
        
        # Lower Red range for testing
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        # Upper Red range
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        # Frame size
        frame_resolution = (640, 480)

        # Check for ordinance template
        resized_template_name = "resized_ordinance_template.png"
        resized_template = None

        if os.path.isfile(resized_template_name.lower()):
            resized_template = cv2.imread(resized_template_name.lower())
        else:
            if os.path.isfile(self.ordinance_template) is not None:
                template = cv2.imread(self.ordinance_template)
                resized_template = cv2.resize(template, frame_resolution, interpolation=cv2.INTER_LINEAR) # Preserve transparency

                # Save the resized image if needed
                cv2.imwrite(resized_template_name.lower(), resized_template)

        # Add camera
        self.pitop.add_component(Camera(format="OpenCV", rotate_angle=90, resolution=frame_resolution))


        def detect_line(frame):
            processed_frame = process_frame_for_line(frame)

            if processed_frame.line_center is not None:
                self.pitop.miniscreen.display_image(processed_frame.robot_view)
                return processed_frame.angle
            else:
                print("\n No line detected.", end="\r")
                return None
    

        def detect_ordinance(frame):
            # Convert BGR to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Apply Gaussian blur to smooth the frame
            blurred_hsv = cv2.GaussianBlur(hsv_frame, (9, 9), 0)
            
            # Threshold to get the mask for the desired color
            # mask = cv2.inRange(blurred_hsv, lower_ordinance_hsv, upper_ordinance_hsv)

            # Create masks for both red ranges
            mask1 = cv2.inRange(blurred_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(blurred_hsv, lower_red2, upper_red2)

            # Combine masks
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Morphological operations to clean the mask
            morph_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            morph_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

            # Find contours
            contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            self.pitop.miniscreen.display_image(contours)
                

        def follow_angle(angle):
            # Proportional speed based on the angle, ensuring the max speed doesn't exceed 1
            drive_speed = max(0.1, min(abs(angle) / 360, 1))  # Scale the speed based on the angle (max speed = 1)
            self.drive("forward", speed=drive_speed, hold=True)  # Move forward with proportional speed
            self.pitop.drive.target_lock_drive_angle(angle)  # Lock target direction based on angle


        def estimate_distance_to_ordinance(ordinance_width, bbox_width):
            KNOWN_WIDTH = ordinance_width
            FOCAL_LENGTH = 420

            if bbox_width < 10 or bbox_width > 500:
                print(f"Invalid bbox width ({bbox_width}), skipping distance calculation.")
                return None

            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width

            if distance < 15 or distance > 40:
                print(f"Distance {distance} cm out of range. Ignoring measurement.")
                return None

            self.distance_history.append(distance)
            return sum(self.distance_history) / len(self.distance_history)
            
        try:
            # Conditionals
            capturing_ordinance = False
            red_square_detected = False
            
            # Parameters
            red_square_count = 0
            initial_claw_angle = -45 # degrees
            initial_camera_angle = 12 # degrees
            ordinance_width = 5.0 # cm
            target_distance = 19  # cm
            
            # Initial Setup
            self.pan_servo_to_angle(self.claw_servo, initial_claw_angle)
            self.pan_servo_to_angle(self.camera_servo, initial_camera_angle)
            
            """
            Logic Process:
            
            1. Detect line
            2. While detecting line, look for red squares
                - Red squares will be what gives our robot a choice
                to make a decision.
            3. When the red square is detected look for ordinances based on
                the location the judges specify.
                - Use the red_squares_count to count squares in phase two, which allow
                the robot to know where to drop off ordinances
            
            This is not the final logic process, there are still changes to be made
            
            """
            
            while self.running:
                frame = self.pitop.camera.get_frame()
                if frame is None:
                    continue

                # 1. Detect the line and red squares
                line_angle = detect_line(frame)
                # TODO red square detection
                
                # 3. Line-following
                if line_angle:  # If line is detected and the angle is significant
                    pass
                
                # 4. TODO Red square detected
                
                ordinance_angle = detect_ordinance(frame)

                # 5. Ordinance logic
                if ordinance_angle and not capturing_ordinance:
                    print(f"Following line. Line angle: {line_angle}")
                    follow_angle(line_angle)
                    
                else:
                    print("No line or ordinance detected.", end="\r")
                    
        except KeyboardInterrupt:
            self.running = False
            print("Stopping autonomous navigation.")


    def __run__(self):
        self.running = True
        if self.navigation_method == "remote":
            self.remote_navigation()
        elif self.navigation_method == "autonomous":
            self.autonomous_navigation()
        
