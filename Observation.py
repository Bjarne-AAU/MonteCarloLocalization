
import pygame
import numpy as np
import scipy.stats as stats

from scipy.ndimage.filters import gaussian_filter

import cv2

from matplotlib import cm

from World import WORLD_TYPE
from World import world_type_from_colormap
from World import create_palette

class ObservationModel(object):

    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    @staticmethod
    def evaluate(robot, world):
        scale = 1
        map = world.map([0,0] + list(world.size + robot.size - 1))
        area = world.map(list(robot.pos) + list(robot.size))

        map_m = pygame.surfarray.array2d(pygame.transform.scale(map, np.array(map.get_size())/scale)).astype(np.float)
        area_m = pygame.surfarray.array2d(pygame.transform.scale(area, np.array(area.get_size())/scale)).astype(np.float)

        randn = np.random.normal(0, 5.0, area_m.shape) * 30 / scale
        area_m += randn

        area_m[area_m > 255.0] = 255.0
        area_m[area_m < 0.0] = 0.0


        res = -cv2.matchTemplate(map_m.astype(np.uint8), area_m.astype(np.uint8), cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        res = (res-min_val)/(max_val-min_val)

        # res = cv2.matchTemplate(map_m.astype(np.uint8), area_m.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # if (max_val > min_val):
        #     res = (res-min_val)/(max_val-min_val)
        # else: res[:] = 1

        # res = cv2.matchTemplate(map_m.astype(np.uint8), area_m.astype(np.uint8), cv2.TM_CCORR_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # res = (res-min_val)/(max_val-min_val)
        # # res = np.power(res, 10)



        res = np.roll(res, robot.width/(2*scale), 0)
        res = np.roll(res, robot.height/(2*scale), 1)

        res = cv2.resize(res, (world.height, world.width), interpolation = cv2.INTER_NEAREST)

        return res
