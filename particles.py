
import numpy as np
from Robot import Robot


class AbstractParticles(object):

    def __init__(self):
        pass



class ParticlesGrid(AbstractParticles):

    def __init__(self, size):
        self._particles = np.zeros(size)


class Particles(AbstractParticles):

    def __init__(self, N, map):
        width = map.width
        height = map.height
        # self._particles = map.

    @property
    def weights(self):
        return self._particles[:,2]

    @property
    def positions(self):
        return self._particles[:,0:2]

    def at(self, index):
        return self._particles[index,:]
