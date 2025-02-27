from PitopRobot import Pitop, PanTiltController
from time import sleep
from signal import pause

def pan_servo_to_angle(servo: PanTiltController, angle: int = 0):
    """
    Pans the servo to a specified angle.

    Parameters:
    servo (PanTiltController): The PanTiltController instance controlling the pan and tilt servos.
    angle (int): The target angle to pan the servo to, default is 0.

    This method adjusts the pan servo's target angle to the specified value.
    """
    servo.pan_servo.target_speed = 100
    servo.pan_servo.target_angle = angle
    
robot = Pitop()
claw_servo = PanTiltController(servo_pan_port="S2")
claw_servo.pan_servo.smooth_acceleration = False


pan_servo_to_angle(claw_servo, -30)
sleep(2)
pan_servo_to_angle(claw_servo, 30)

pause()