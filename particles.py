
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
        self.N = N
        width = map.width
        height = map.height
        self._particles = np.hstack((np.random.randint(0, width, (N,1)), np.random.randint(0, height, (N,1))))

    @property
    def weights(self):
        return self._particles[:,2]

    @property
    def positions(self):
        return self._particles[:,0:2]

    @positions.setter
    def positions(self, positions):
        self._particles[:,0:2] = positions

    def at(self, index):
        return self._particles[index,:]
