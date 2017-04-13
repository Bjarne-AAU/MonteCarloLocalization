
import numpy as np
import pygame
import cv2

from scipy import linalg
from scipy import stats

from Sensor import Sensor as AbstractSensor
from Sensor import SensorNoise as AbstractSensorNoise

class Sensor(AbstractSensor):

    def __init__(self, fps = 10):
        super(Sensor, self).__init__(fps)
        self._last = None

    def _observe(self, what):
        if self._last is None: self._last = what
        motion = what - self._last
        self._last = what
        return motion

    def _draw(self, image):
        pos = np.mod(self._last - self.observation, image.get_size())
        pygame.draw.line(image, (255,0,0), pos, self._last + 10 * self.observation, 2)


class SensorNoise(AbstractSensorNoise):
    pass

class SensorNoiseGaussian(SensorNoise):

    def create(self, motion):
        level_dist, level_rot = self.level
        mag = linalg.norm(motion) + 1.0
        return stats.multivariate_normal.rvs((0,0), mag*level_dist/5.0)


class SensorNoiseAdvanced(SensorNoise):

    def create(self, motion):
        level_dist, level_rot = self.level
        vec = stats.multivariate_normal.rvs(motion, level_dist/5.0)

        mag = np.maximum(linalg.norm(vec), 0.0001 )

        # create noise
        noise_rot = stats.norm.rvs(0, level_rot*level_rot/2.0)
        noise_dist = stats.norm.rvs(0, level_dist/2.0)

        # rotate direction vector
        sin_angle = np.sin(noise_rot)
        cos_angle = np.cos(noise_rot)
        dir_x = vec[0] * cos_angle - vec[1] * sin_angle
        dir_y = vec[0] * sin_angle + vec[1] * cos_angle
        vec = np.array( (dir_x, dir_y) )

        vec += vec * noise_dist
        return vec - motion


class SensorNoiseModel(object):
    GAUSSIAN  = SensorNoiseGaussian
    ADVANCED = SensorNoiseAdvanced





class Observation(object):

    @classmethod
    def evaluate(cls, observation, kernel_size):
        motions = np.vstack(np.indices(kernel_size).transpose(1,2,0)) - np.array(kernel_size)/2 + observation
        kernel = cls.evaluate_at(motions, observation).reshape(kernel_size)
        kernel = cv2.GaussianBlur(kernel, (15,15), 1.0)
        kernel /= np.sum(kernel)
        return kernel

    @classmethod
    def evaluate_at(cls, motions, observation):
        raise NotImplementedError()

class ObservationGaussian(Observation):

    @classmethod
    def evaluate_at(cls, motions, observation):
        probs = linalg.norm(motions - observation, axis=1)
        return np.exp(-probs)

class ObservationAdvanced(Observation):

    level_dist = 0.2
    level_rot = 0.5

    @classmethod
    def evaluate_at(cls, motions, observation):
        probs = np.zeros(len(motions))

        mag = np.maximum(linalg.norm(observation), 0.001)
        mags = linalg.norm(motions, axis=1)
        indices = mags.nonzero()[0]

        probs += stats.norm.logpdf( mags/mag, 1, cls.level_dist/2.0)

        angles = np.ones(len(motions))
        angles[indices] = observation.dot(motions[indices,:].T) / (mags[indices] * mag)
        probs += stats.norm.logpdf(angles, 1, cls.level_rot*cls.level_rot/2.0)

        return np.exp(probs)


class ObservationModel(object):
    GAUSSIAN = ObservationGaussian
    ADVANCED = ObservationAdvanced



