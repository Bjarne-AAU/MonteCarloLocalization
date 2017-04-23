
import numpy as np
import pygame
import cv2

from scipy import linalg
from scipy import stats

from Tools import as_int

from Sensor import Sensor as AbstractSensor
from Sensor import SensorNoise as AbstractSensorNoise

class Sensor(AbstractSensor):

    def __init__(self, size, fps=None):
        super(Sensor, self).__init__(fps)
        self._size = np.array(size)

    def _observe(self, what):
        location, world = what
        location %= world.size
        image = world.map(np.concatenate((location - self._size/2, self._size)))
        return image.copy()

    def _draw(self, image):
        view = pygame.transform.scale(self.observation, image.get_size())
        image.blit(view, (0, 0))



class SensorNoise(AbstractSensorNoise):

    def prior(self, x):
        d1 = stats.cauchy(90, 50)
        d2 = stats.cauchy(30, 100)
        res = d1.pdf(x) + d2.pdf(x)
        return res/(d1.pdf(90) + d2.pdf(90))

    def add(self, observation):
        res = observation.copy()
        mat = pygame.surfarray.pixels2d(res)
        tmp = mat + self.create(mat)
        # tmp = cv2.GaussianBlur(tmp, (5,5), 2)
        mat[:] = np.clip(tmp, 0, 255)
        return res

class SensorNoiseGaussian(SensorNoise):

    def create(self, mat):
        prior = self.prior(mat)
        sigma = prior * self.level * self.level * 100
        return np.random.normal(0, sigma, mat.shape)


class SensorNoiseSalt(SensorNoise):

    def create(self, mat):
        prior = self.prior(mat)
        N = as_int(mat.size * self.level * self.level * 0.25)
        res = np.zeros(mat.shape, dtype=np.int)
        ind = np.random.choice(res.size, N, p=(prior/np.sum(prior)).ravel())
        pos = np.unravel_index(ind, res.shape)
        res[pos] = 255
        return res

class SensorNoisePepper(SensorNoise):

    def create(self, mat):
        prior = self.prior(mat)
        N = as_int(mat.size * self.level * self.level * 0.25)
        res = np.zeros(mat.shape, dtype=np.int)
        ind = np.random.choice(res.size, N, p=(prior/np.sum(prior)).ravel())
        pos = np.unravel_index(ind, res.shape)
        res[pos] = -255
        return res

class SensorNoiseSaltPepper(SensorNoise):

    def create(self, mat):
        prior = self.prior(mat)
        N = as_int(mat.size * self.level * self.level * 0.25)
        res = np.zeros(mat.shape, dtype=np.int)
        ind = np.random.choice(res.size, N, p=(prior/np.sum(prior)).ravel())
        pos = np.unravel_index(ind, res.shape)
        res[pos] = (np.random.randint(0, 2, N)*2-1) * 255
        return res


class SensorNoiseSpeckle(SensorNoise):

    def create(self, mat):
        prior = self.prior(mat)
        sigma = np.sqrt(prior) * self.level * self.level * 2
        noise = np.random.normal(0, sigma, mat.shape)
        res = (135.0 - np.abs(mat-120.0)) * noise
        return res


class SensorNoiseModel(object):
    GAUSSIAN    = SensorNoiseGaussian
    SALT        = SensorNoiseSalt
    PEPPER      = SensorNoisePepper
    SALT_PEPPER = SensorNoiseSaltPepper
    SPECKLE     = SensorNoiseSpeckle





class Observation(object):

    @classmethod
    def evaluate(cls, observation, world, locations=None):
        pos = -np.array(observation.get_size())/2
        size = world.size + observation.get_size() - 1
        model = world.map(np.concatenate((pos, size)))
        model_m = pygame.surfarray.pixels2d(model).astype(np.float32)
        observation_m = pygame.surfarray.pixels2d(observation).astype(np.float32)
        if locations is None:
            res = cls.evaluate_global(model_m, observation_m)
        else:
            res = cls.evaluate_local(locations, model_m, observation_m)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if (max_val > min_val):
            res = (res-min_val)/(max_val-min_val)
        else: res = res/max_val
        return res

    @classmethod
    def evaluate_global(cls, world, observation):
        raise NotImplementedError()

    @classmethod
    def evaluate_local(cls, locations, world, observation):
        return np.apply_along_axis(cls.evaluate_at, 1, locations, world, observation)

    @classmethod
    def evaluate_at(cls, location, world, observation):
        raise NotImplementedError()

    @classmethod
    def _extract_location(cls, location, size, world):
        start = location
        end = start + size
        return world[ start[0]:end[0], start[1]:end[1] ]


class ObservationMDIFF(Observation):

    @classmethod
    def evaluate_global(cls, world, observation):
        x,y = observation.shape
        model = cv2.blur(world, observation.shape)
        model = model[x/2:-x/2+1, y/2:-y/2+1]
        return -np.abs(model - np.mean(observation))

    @classmethod
    def evaluate_local(cls, locations, world, observation):
        x,y = observation.shape
        model = cv2.blur(world, observation.shape)
        model = model[x/2:-x/2+1, y/2:-y/2+1]
        observation_mean = np.mean(observation)
        return np.apply_along_axis(cls.evaluate_at, 1, locations, model, observation, observation_mean)

    @classmethod
    def evaluate_at(cls, location, world, observation, observation_mean):
        pos = tuple(location)
        return -np.abs(world[pos] - observation_mean)


class ObservationSQDIFF(Observation):

    @classmethod
    def evaluate_global(cls, world, observation):
        return -cv2.matchTemplate(world, observation, cv2.TM_SQDIFF)

    @classmethod
    def evaluate_at(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        diff = observation - model
        return -np.sum(diff*diff)
        # return -np.linalg.norm(model - observation)


class ObservationCCOEFF(Observation):

    @classmethod
    def evaluate_global(cls, world, observation):
        return cv2.matchTemplate(observation, world, cv2.TM_CCOEFF_NORMED)

    @classmethod
    def evaluate_at(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        norm = np.sqrt(np.sum(model*model) * np.sum(observation*observation))
        return np.sum( (model - np.mean(model)) * (observation - np.mean(observation)) ) / norm
        # return np.sum( (model - np.mean(model)) * (observation - np.mean(observation)) )


class ObservationCCORR(Observation):

    @classmethod
    def evaluate_global(cls, world, observation):
        r = cv2.matchTemplate(world, observation, cv2.TM_CCORR_NORMED)
        return ((r**2)**2)**2

    @classmethod
    def evaluate_at(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        norm = np.sqrt(np.sum(model*model) * np.sum(observation*observation))
        r = np.sum( model * observation ) / norm
        return ((r**2)**2)**2


class ObservationModel(object):
    SQDIFF = ObservationSQDIFF
    CCOEFF = ObservationCCOEFF
    MDIFF = ObservationMDIFF
    CCORR  = ObservationCCORR
