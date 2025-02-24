from pitop import Pitop, Camera

robot = Pitop()
robot.add_component(Camera(format="OpenCV", rotate_angle=90))

while True:
    robot.miniscreen.display_image(robot.camera.get_frame())