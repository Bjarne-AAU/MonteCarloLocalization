
import numpy as np
import pygame
import cv2

from Tools import as_int


class TYPE(object):
    DENSITY = 0
    LIKELIHOOD = 1
    KERNEL = 2

class ESTIMATE(object):
    MAP = 0
    MLE = 1
    EXP = 2

class Representation(object):

    def __init__(self, size):
        self.N = size[0]*size[1]
        self.size = size
        self._kernel = np.ones( (35, 35) )
        self._locations = np.vstack(np.indices(size).transpose(1,2,0)).astype(np.float)
        self._likelihood = np.ones( size ) / self.N
        self._density = np.ones( size ) / self.N

    def reset(self):
        self._likelihood[:] = 1.0 / self.N
        self._density[:] = 1.0 / self.N

    @property
    def locations(self):
        return as_int(self._locations) % self.size

    def estimate(self, estimate_type):
        if estimate_type == ESTIMATE.MAP:
            return self._locations[np.argmax(self._density)]
        elif estimate_type == ESTIMATE.MLE:
            return self._locations[np.argmax(self._likelihood)]
        elif estimate_type == ESTIMATE.EXP:
            mean = np.average(self._locations, weights=self._density.ravel(), axis=0)
            #var  = np.average(np.square(self._locations - mean), weights=self._density.ravel(), axis=0)
            return mean

    def propagate(self, motion, motion_model):
        raise NotImplementedError()

    def update(self, view, model, observation_model):
        raise NotImplementedError()

    def _draw(self, image, data):
        if image.get_size() != data.shape:
            data = cv2.resize(data, image.get_size())
        mat = pygame.surfarray.pixels3d(image)
        mat[:,:,0] = mat[:,:,0] * (data * 0.99 + 0.01)
        mat[:,:,1] = mat[:,:,1] * (data * 0.99 + 0.01)
        mat[:,:,2] = mat[:,:,2] * (data * 0.99 + 0.01)
        del mat

    def draw(self, image, data_type):
        if data_type == TYPE.DENSITY: data = self._density
        elif data_type == TYPE.LIKELIHOOD: data = self._likelihood
        elif data_type == TYPE.KERNEL: data = self._kernel
        else: return

        self._draw(image, data/np.max(data))

    def draw_estimate(self, image, type):
        est = self.estimate(type)
        pygame.draw.circle(image, (255,0,0), as_int(est), 10, 3)


class Grid(Representation):

    def __init__(self, size):
        super(Grid, self).__init__(size)

    def propagate(self, motion, motion_model):
        self._kernel = motion_model.evaluate(motion, self._kernel.shape)
        self._density = cv2.filter2D(self._density, -1, np.flipud(np.fliplr(self._kernel)), borderType=cv2.BORDER_WRAP)
        self._density = np.roll(self._density, shift=np.round(motion).astype(np.int), axis=(0,1))
        self._density /= np.sum(self._density)

    def update(self, view, model, observation_model):
        self._likelihood = observation_model.evaluate(view, model)
        self._density *= self._likelihood
        self._density += 1.e-15
        self._density /= np.sum(self._density)



class Particles(Representation):

    def __init__(self, N, size):
        super(Particles, self).__init__((N,1))
        self.size = size
        self.reset()

    def reset(self):
        super(Particles, self).reset()
        self._locations[:,0] = np.random.randint(0, self.size[0], self.N)
        self._locations[:,1] = np.random.randint(0, self.size[1], self.N)

    @property
    def N_effective(self):
        density = self._density / np.sum(self._density)
        return 1.0 / np.sum(density * density)

    def propagate(self, motion, motion_model):
        if self.N_effective < 0.5 * self.N:
            # self.resample_naive()
            # self.resample_stratified()
            self.resample_systematic()

        self._kernel = motion_model.evaluate(motion, self._kernel.shape)
        indices = np.random.choice(self._kernel.size, len(self._locations), p=self._kernel.ravel())
        self._locations += np.array(np.unravel_index(indices, self._kernel.shape)).T + 0.5
        self._locations -= np.array(self._kernel.shape)/2.0
        self._locations += motion
        self._locations %= self.size

    def update(self, observation, model, observation_model):
        self._likelihood[:,0] = observation_model.evaluate(observation, model, self.locations)
        self._density *= self._likelihood
        self._density += 1.e-15
        self._density /= np.sum(self._density)


    def draw(self, image, type):
        if type == TYPE.KERNEL:
            self._draw(image, self._kernel/np.max(self._kernel))
        else:
            if type == TYPE.DENSITY: weights = self._density
            elif type == TYPE.LIKELIHOOD: weights = self._likelihood

            for pos, w in zip(self.locations, weights/np.max(weights)):
                pygame.draw.circle(image, (255,0,0), pos, as_int(w*5))


    def resample_naive(self):
        positions = np.random.random(self.N)
        self._resample_indices(positions)

    def resample_stratified(self):
        positions = (np.arange(self.N) + np.random.random(self.N)) / self.N
        self._resample_indices(positions)

    def resample_systematic(self):
        positions = (np.arange(self.N) + np.random.random()) / self.N
        self._resample_indices(positions)

    def _resample_indices(self, positions):
        indices = np.searchsorted(np.cumsum(self._density), positions, side='right')
        self._locations = self._locations[indices]
        self._density = self._density[indices]
        self._density /= np.sum(self._density)
        # self._density[:] = 1.0/self.N
