
import numpy as np
import cv2
import pygame

from scipy import linalg
from scipy import stats


class MotionSensor(object):

    def __init__(self, state):
        self._last = state
        self._motion = np.zeros(self._last.shape)

    @property
    def observation(self):
        return self._motion

    def observe(self, state):
        self._motion = state - self._last
        self._last = state

    def add_noise(self, noisemodel, level=0.5):
        noisemodel.add(self._motion, level, level)

    # def draw(self, image):
    #     pos = np.mod(self._last[0:2] - self._motion[0:2], image.get_size())
    #     # pos -= self.size/2
    #     pygame.draw.line(image, (255,0,0), pos, self._last[0:2]*10, 2)


class MotionSensorNoise(object):

    @staticmethod
    def add(motion, level_pos, level_rot):
        raise NotImplementedError()

    @staticmethod
    def prob(motion, motion_ref, level_pos, level_rot):
        raise NotImplementedError()


class MotionSensorNoiseGaussian(MotionSensorNoise):

    @staticmethod
    def add(motion, level_dist, level_rot):
        dist = linalg.norm(motion[0:2])
        motion[0:2] *= stats.norm.rvs(1, level_dist)
        motion[2] += stats.norm.rvs(0, level_rot)

    @staticmethod
    def prob(motions, motion_ref, level_dist, level_rot):
        vec = motion_ref[0:2]
        vecs = motions.T[0:2,]
        mag = linalg.norm(vec)
        mags = linalg.norm(vecs, axis=0)

        sigma = max(mag, 0.5)*level_dist/5.0
        probs = stats.norm.pdf(mags, mag, sigma)

        norm = mags * mag
        indices = norm.nonzero()[0]
        angles = np.ones(len(motions))
        angles[indices] = vec.dot(vecs[:,indices]) / norm[indices]
        # probs = probs * stats.norm.pdf(angles, 1, level_rot*level_rot/2.0)

        return probs/np.max(probs)

        # only positional data is considered
        # rot = motions.T[2,] - motion_ref[2]
        # rot_p = stats.norm.pdf(rot, 1, level_rot)


class MotionSensorNoiseModel(object):
    GAUSSIAN  = MotionSensorNoiseGaussian


class Motion(object):


    @classmethod
    def kernel(cls, density, motion, noisemodel):
        kernel_size = 49
        motions = -np.vstack(np.mgrid[-kernel_size/2:kernel_size/2+1, -kernel_size/2:kernel_size/2+1].T) + motion[0:2]
        kernel = noisemodel.prob(motions, motion, 2.0, 0.6).reshape((kernel_size+1, kernel_size+1)).T
        return kernel

    @classmethod
    def propagate_density(cls, density, motion, noisemodel):

        kernel_size = 49
        motions = -np.vstack(np.mgrid[-kernel_size/2:kernel_size/2+1, -kernel_size/2:kernel_size/2+1].T) + motion[0:2]
        kernel = noisemodel.prob(motions, motion, 2.0, 0.6).reshape((kernel_size+1, kernel_size+1)).T

        density = cv2.filter2D(density, -1, kernel, borderType=cv2.BORDER_WRAP)
        density = np.roll(density, motion[0:2].astype(np.int), axis=(0,1))

        return density

    @classmethod
    def propagate_global(cls, world, observation):
        raise NotImplementedError()

    @classmethod
    def propagate_local(cls, locations, world, observation):
        return np.apply_along_axis(cls.evaluate_at, 1, locations, world, observation)

    @classmethod
    def propagate_at(cls, location, world, observation):
        raise NotImplementedError()



class MotionModel(object):

    @staticmethod
    def evaluate(motion, particles):
        vel = motion[0:2]
        rot = motion[2:3]

        def noiseFun(pos):
            # print(pos)
            n = np.random.normal(0, 2, 2)
            return pos + vel + n

        particles.positions = np.apply_along_axis(noiseFun, 1, particles.positions)

        return particles

