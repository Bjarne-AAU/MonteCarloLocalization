
import pygame
import numpy as np
import scipy.stats as stats

class MotionModel(object):

    @staticmethod
    def evaluate(motion, particles):
        vel = motion[0:2]
        rot = motion[2:3]

        def noiseFun(pos):
            # print(pos)
            return pos + vel

        particles.positions = np.apply_along_axis(noiseFun, 1, particles.positions)

        return particles

