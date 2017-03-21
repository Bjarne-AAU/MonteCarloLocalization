
import numpy as np
from Robot import Robot


class ParticleGrid(object):

    def __init__(self, size):
        self._particles = np.zeros(size)

