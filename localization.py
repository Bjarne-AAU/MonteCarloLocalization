import sys

import numpy as np
import scipy
import matplotlib

import pygame
import noise
import cv2

print("Version check")
print("=============")
print("python    : {}".format(sys.version.split(' ')[0]))
print("numpy     : {}".format(np.__version__))
print("scipy     : {}".format(scipy.__version__))
print("matplotlib: {}".format(matplotlib.__version__))
print("pygame    : {}".format(pygame.__version__))
print("noise     : {}".format(noise.__version__))
print("cv2       : {}".format(cv2.__version__))
print("=============")

pygame.init()


from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import generic_filter

from gui.window import MainWindow

from World import NOISE
from World import WORLD_TYPE
from World import WorldMap

from matplotlib import cm

from World import world_type_from_colormap
from World import create_palette

from Robot import Robot

import Vision
import Motion
import Representation

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=150)

def to_int(text):
    try:
        return int(float(text))
    except:
        return None


win_width = 1000
win_height = 600

map_seed = np.random.randint(1000)
map_seed = 3
map_width = win_width-200
map_height = win_height
robot_width = 20
robot_height = 20
robot_pos_x = map_width/2
robot_pos_y = map_height/2
robot_image = "gui/rocket.png"


win = MainWindow(win_width, win_height, "Monte Carlo Localization - Demo application")
win.FPS = 30
win.set_option("seed", map_seed)

vs = (15,15)

vision = Vision.Sensor(vs)
motion = Motion.Sensor()

# OPTIONS for noise: GAUSSIAN, SALT, PEPPER, SALT_PEPPER, SPECKLE
# OPTIONS for model: MDIFF, SQDIFF, CCORR, CCOEFF
vision_sensor = Vision.Sensor(vs, fps=5)
vision_sensor_noise = Vision.SensorNoiseModel.GAUSSIAN(0.5)
vision_sensor.set_noise_model(vision_sensor_noise)
vision_model = Vision.ObservationModel.MDIFF

# OPTIONS for noise: GAUSSIAN, ADVANCED
# OPTIONS for model: GAUSSIAN, ADVANCED
motion_sensor = Motion.Sensor(fps = 15)
motion_sensor_noise = Motion.SensorNoiseModel.ADVANCED( (0.3,0.5) )
motion_sensor.set_noise_model(motion_sensor_noise)
motion_model = Motion.ObservationModel.ADVANCED

# REPRESENTATION
# density = Representation.Grid( (map_width, map_height) )
density = Representation.Particles( 1000, (map_width, map_height) )


clock = pygame.time.Clock()
while win.running:
    events = win.get_events()

    dt = clock.tick()/1000.0

    keys = np.array(pygame.key.get_pressed())
    keys = np.where(keys)[0]
    for key in keys:
        if key == pygame.K_UP: robot.accelerate(-70, dt)
        if key == pygame.K_DOWN: robot.accelerate(70, dt)
        if key == pygame.K_LEFT: robot.rotate(200, dt)
        if key == pygame.K_RIGHT: robot.rotate(-200, dt)


    if win.get_option("generate"):
        win.reset_option("generate")
        seed = to_int(win.get_option("seed"))
        if seed is None: print("No valid seed")
        world_type = WORLD_TYPE.get(win.get_option("world"))
        if world_type is None: print("No valid world type")
        if seed is not None and world_type is not None:
            print("Generate {} with seed {}".format(win.get_option("world"), seed))
            world = WorldMap(map_width, map_height, pad_x=40, pad_y=40, scale=8, colors=world_type, type=NOISE.SIMPLEX, seed=seed)
            robot = Robot(robot_pos_x, robot_pos_y, 0, robot_width, robot_height, robot_image)
            density.reset()

    if win.get_option("reset"):
        density.reset()
        win.reset_option("reset")


    is_active = win.get_option("start") or win.get_option("step")

    # Move robot
    robot.update(dt)

    # Observation for display
    vision.observe( (robot.pos, world) )
    motion.observe( robot.pos )

    # Observe motion and propagate density/particles accordingly
    if is_active:
        if motion_sensor.observe( robot.pos ):
            density.propagate(motion_sensor.observation, motion_model)

    # Observe view and update density/particles accordingly
    if vision_sensor.observe( (robot.pos, world) ):
        if is_active:
            density.update(vision_sensor.observation, world, vision_model)

    win.reset_option("step")

    # Draw
    mainmap = win.get_mainmap_canvas()

    world.draw(mainmap)

    if win.get_option("show_posterior"):
        density.draw(mainmap, Representation.TYPE.DENSITY)
        if win.get_option("show_estimate"):
            density.draw_estimate(mainmap, Representation.ESTIMATE.MAP)
    elif win.get_option("show_likelihood"):
        density.draw(mainmap, Representation.TYPE.LIKELIHOOD)
        if win.get_option("show_estimate"):
            density.draw_estimate(mainmap, Representation.ESTIMATE.MLE)

    if not win.get_option("hide_robot"):
        robot.scale = 1.0
        robot.set_color(255, 80, 80, 255)
        robot.draw(mainmap)


    minimap = win.get_minimap_canvas()

    if is_active:
        # motion_sensor.draw(mainmap)
        vision_sensor.draw(minimap)
    else:
        # motion.draw(mainmap)
        vision.draw(minimap)

    if win.get_option("show_kernel"):
        density.draw(minimap, Representation.TYPE.KERNEL)

    robot.scale = 5.0
    robot.set_color(0,0,0,80)
    robot.draw(minimap, minimap.get_rect().center)

    win.update()


