
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

    def sample(self):  #implement non-stratified sampling
        sampled_particles = np.zeros(self._particles.shape)  #empty array
        draws = np.random.uniform(low=0.0, high=1.0, size=self.N)
        cdf = np.cumsum(self.weights)
        cdf = cdf/cdf[-1] #normalizing cdf
        for i in range(self.N):
            j=0
            while draws[i] > cdf[j]:
                j += 1
            sampled_particles[i] = self._particles[j]
        self._particles = sampled_particles

    def sample_stratified(self): #implement stratified sampling
        N = len(self.weights)
        # make N subdivisions, and choose a random position within each one
        positions = (random(N) + range(N)) / N
        sampled_particles = np.zeros(self._particles.shape)  #empty array store resampling
        cdf = np.cumsum(self.weights)
        cdf = cdf/cdf[-1] #normalizing cdf
        i, j = 0, 0
        while i < N:
            if positions[i] < cdf[j]:
                sampled_particles[i] = j
                i += 1
            else:
                j += 1
        self._particles = sampled_particles


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
