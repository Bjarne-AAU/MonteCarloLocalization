import sys
import pygame
pygame.init()


from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import generic_filter
import numpy as np
import cv2

from gui.window import MainWindow

from World import NOISE
from World import WORLD_TYPE
from World import WorldMap

from matplotlib import cm

from World import world_type_from_colormap
from World import create_palette

from Robot import Robot

from Observation import VisionSensor
from Observation import VisionSensorNoiseModel
from Observation import ObservationModel

from Motion import MotionSensor
from Motion import MotionSensorNoiseModel
from Motion import MotionModel
from particles import *

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
map_width = win_width-200
map_height = win_height
robot_width = 20
robot_height = 20
robot_pos_x = map_width/2
robot_pos_y = map_height/2
robot_image = "gui/rocket.png"

N_particles = 500
sample_threshold = N_particles*0.40

#old_pos = np.array( (robot_state, robot_pos_y) )
robot_state = np.array([robot_pos_x, robot_pos_y, 0])

win = MainWindow(win_width, win_height, "Monte Carlo Localization - Demo application")
win.FPS = 30
win.set_option("seed", map_seed)


vision_sensor = VisionSensor((robot_width, robot_height))
motion_sensor = MotionSensor(robot_state)


like_m = None
posterior_m = None

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
            particles = Particles(N_particles,world)


    mainmap = win.get_mainmap_canvas()
    world.draw(mainmap)

    robot.update(dt)
    motion_sensor.observe(robot.state)
    motion_sensor.add_noise(MotionSensorNoiseModel.GAUSSIAN, 0.5)
    # motion_sensor.draw(mainmap)


    robot_motion = robot.state - robot_state

    MLE = None
    MAP = None

    if win.get_option("reset"):
        particles = Particles(N_particles,world)
        posterior_m = None
        like_m = None
        win.reset_option("reset")


    if win.get_option("start") or win.get_option("step"):
        particles = MotionModel.evaluate(robot_motion, particles)
        if not win.get_option("particle_posterior"):
            like_m = ObservationModel.MDIFF.evaluate(vision_sensor.observation, world)
            if posterior_m is None: posterior_m = np.ones(like_m.shape, dtype=np.float)

            kx = int(dt * np.sqrt(robot_motion[1] * robot_motion[1]) * 15.0) * 2 + 3
            ky = int(dt * np.sqrt(robot_motion[0] * robot_motion[0]) * 15.0) * 2 + 3
            posterior_m = cv2.GaussianBlur(posterior_m, (kx, ky), cv2.BORDER_WRAP)

            posterior_m = np.roll(posterior_m, int(round(robot_motion[0])), 0)
            posterior_m = np.roll(posterior_m, int(round(robot_motion[1])), 1)

            MLE = np.unravel_index(like_m.argmax(), like_m.shape)
            posterior_m *= like_m
            #posterior_m = np.exp(posterior_m - np.max(posterior_m))
            posterior_m = posterior_m/np.max(posterior_m)

            MAP = np.unravel_index(posterior_m.argmax(), posterior_m.shape)
            if like_m is not None and win.get_option("show_likelihood"):
                like = pygame.surfarray.make_surface((like_m * 240 + 15))
                like.set_palette(create_palette(cm.gray))
                mainmap.blit(like, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            if posterior_m is not None and win.get_option("show_posterior"):
                posterior = pygame.surfarray.make_surface((posterior_m * 240 + 15))
                posterior.set_palette(create_palette(cm.gray))
                mainmap.blit(posterior, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        else:
            like_m = ObservationModel.MDIFF.evaluate(vision_sensor.observation, world, particles.positions)

            if posterior_m is None: posterior_m = np.ones(like_m.shape, dtype=np.float)

            posterior_m *= like_m
            posterior_m += 0.01

            posterior_m = posterior_m/np.max(posterior_m)
            particles.weights = posterior_m
            index = posterior_m.argmax()
            MAP = (particles.at(index)[0:2]).astype(np.int)
            if particles.effectiveN < sample_threshold:
                particles.resample()
                # posterior_m[:] = 1.0

        robot_state = robot.state
        win.reset_option("step")





    if posterior_m is not None and win.get_option("draw_particle"):
       # print particles.positions.shape
        for n in range(particles.N):
            particle_x = particles.at(n)[0]
            particle_y = particles.at(n)[1]
            pygame.draw.circle(mainmap, (255,0,0), np.round(np.array([particle_x,particle_y])).astype(np.int),
                np.round(posterior_m[n]*10).astype(np.int))





    if MLE is not None:
        pygame.draw.circle(mainmap, (255,0,0), MLE, 5, 2)

    if MAP is not None:
        pygame.draw.circle(mainmap, (0,255,0), MAP, 10, 2)


    if not win.get_option("hide_robot"):
        robot.scale = 1.0
        robot.set_color(255, 80, 80, 255)
        robot.draw(mainmap)


    vision_sensor.observe(robot.pos, world)
    # vision_sensor.add_noise(VisionSensorNoiseModel.SALT_PEPPER, 0.6)
    vision_sensor.add_noise(VisionSensorNoiseModel.SPECKLE, 0.7)
    # vision_sensor.add_noise(VisionSensorNoiseModel.GAUSSIAN, 0.5)

    minimap = win.get_minimap_canvas()

    vision_sensor.draw(minimap)

    robot.scale = 5.0
    robot.set_color(0,0,0,80)
    robot.draw(minimap, minimap.get_rect().center)



    win.update()


