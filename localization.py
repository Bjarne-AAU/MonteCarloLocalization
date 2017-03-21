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

from Observation import ObservationModel

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=150)

def to_int(text):
    try:
        return int(float(text))
    except:
        return None


win_width = 1000
win_height = 600

map_seed = np.random.randint(10000)
map_width = win_width-200
map_height = win_height
robot_width = 20
robot_height = 20
robot_pos_x = map_width/2
robot_pos_y = map_height/2
robot_image = "gui/rocket.png"

old_pos = np.array( (robot_pos_x, robot_pos_y) )

win = MainWindow(win_width, win_height, "Monte Carlo Localization - Demo application")
win.FPS = 30
win.set_option("seed", map_seed)


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


    mainmap = win.get_mainmap_canvas()
    mainmap.blit(world.map(), (0, 0))


    robot.update(dt)
    robot_motion = robot.pos - old_pos
    robot.x = robot.x % world.width
    robot.y = robot.y % world.height
    robot.angle = robot.angle % 360

    MLE = None
    MAP = None

    if win.get_option("reset"):
        posterior_m = None
        like_m = None
        win.reset_option("reset")


    if win.get_option("start") or win.get_option("step"):

        like_m = ObservationModel.evaluate(robot, world)
        if posterior_m is None: posterior_m = np.ones(like_m.shape, dtype=np.float)

        kx = int(dt * np.sqrt(robot_motion[1] * robot_motion[1]) * 15.0) * 2 + 3
        ky = int(dt * np.sqrt(robot_motion[0] * robot_motion[0]) * 15.0) * 2 + 3
        posterior_m = cv2.GaussianBlur(posterior_m, (kx, ky), cv2.BORDER_WRAP)

        posterior_m = np.roll(posterior_m, int(round(robot_motion[0])), 0)
        posterior_m = np.roll(posterior_m, int(round(robot_motion[1])), 1)

        MLE = np.unravel_index(like_m.argmax(), like_m.shape)
        posterior_m *= like_m
        posterior_m = posterior_m/np.max(posterior_m)

        MAP = np.unravel_index(posterior_m.argmax(), posterior_m.shape)

        old_pos = robot.pos
        win.reset_option("step")



    if like_m is not None and win.get_option("show_likelihood"):
        like = pygame.surfarray.make_surface((like_m * 240 + 15))
        like.set_palette(create_palette(cm.gray))
        mainmap.blit(like, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    if posterior_m is not None and win.get_option("show_posterior"):
        posterior = pygame.surfarray.make_surface((posterior_m * 240 + 15))
        posterior.set_palette(create_palette(cm.gray))
        mainmap.blit(posterior, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    if MLE is not None:
        pygame.draw.circle(mainmap, (255,0,0), MLE, 5, 2)

    if MAP is not None:
        pygame.draw.circle(mainmap, (0,255,0), MAP, 10, 2)


    if not win.get_option("hide_robot"):
        robot.set_color(255, 80, 80, 255)

        dimg = robot.image
        d, r = divmod(robot.pos + robot.size, world.size)
        if d[0] > 0.0 and r[0] > 0.0: mainmap.blit(dimg, [r[0]-robot.width, robot.y])
        if d[1] > 0.0 and r[1] > 0.0: mainmap.blit(dimg, [robot.x, r[1]-robot.height])
        if d[0] > 0.0 and d[1] > 0.0 and r[0] > 0.0 and r[1] > 0.0: mainmap.blit(dimg, [r[0]-robot.width, r[1]-robot.height])
        mainmap.blit(dimg, robot.pos)


    minimap = win.get_minimap_canvas()
    view = world.map(list(robot.pos) + list(robot.size), minimap.get_size())

    view_m = pygame.surfarray.pixels2d(view)
    randn = np.random.normal(0, 0.1, view.get_size())*255
    randn = gaussian_filter(randn, sigma=5, mode="wrap")
    view_m -= randn
    del view_m

    minimap.blit(view, (0, 0))


    robot.scale = 5.0
    robot.set_color(0,0,0,80)
    center = np.array(minimap.get_rect().center)
    minimap.blit(robot.image, center - robot.size/2.0)
    robot.scale = 1.0



    win.update()


