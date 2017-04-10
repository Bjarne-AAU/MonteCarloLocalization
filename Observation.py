
import numpy as np
import pygame
import cv2

import scipy.stats as stats


class VisionSensor(object):

    def __init__(self, size):
        self._size = np.array(size)
        self._image = pygame.Surface(size, 0, 8)

    @property
    def size(self):
        return self._size

    @property
    def observation(self):
        return self._image

    def observe(self, location, world):
        location = np.mod(location, world.size)
        img = world.map(np.concatenate((location - self._size/2, self._size)))
        self._image.set_palette(img.get_palette())
        self._image.blit(img, (0,0))

    def add_noise(self, noisemodel, level=0.5):
        mat = pygame.surfarray.pixels2d(self._image)
        noisemodel.add(mat, level)

    def draw(self, image):
        view = pygame.transform.scale(self._image, image.get_size())
        image.blit(view, (0, 0))



class VisionSensorNoise(object):

    @staticmethod
    def prior(x):
        d = stats.cauchy(90, 100)
        return d.pdf(x) / d.pdf(0)

    @staticmethod
    def add(mat, level):
        raise NotImplementedError()


class VisionSensorNoiseGaussian(VisionSensorNoise):

    @staticmethod
    def add(mat, level):
        prior = VisionSensorNoise.prior(mat)
        sigma = prior * level * level * 127
        noise = np.random.normal(0, sigma, mat.shape)
        res = mat + noise
        res = cv2.GaussianBlur(res, (5,5), 2)
        mat[:] = np.clip(res, 0, 255)


class VisionSensorNoiseSaltPepper(VisionSensorNoise):

    @staticmethod
    def add(mat, level):
        probs = VisionSensorNoise.prior(mat)
        probs /= np.sum(probs)
        N = mat.size * level * level * 0.25

        res = mat.astype(np.float)
        ind = np.random.choice(res.size, N, p=probs.ravel())
        pos = np.unravel_index(ind, res.shape)
        res[pos] = np.random.randint(0, 2, N) * 255
        res[:] = cv2.GaussianBlur(res, (5,5), 2)
        mat[:] = np.clip(res, 0, 255)


class VisionSensorNoiseSpeckle(VisionSensorNoise):

    @staticmethod
    def add(mat, level):
        prior = VisionSensorNoise.prior(mat)
        sigma = prior * level * level * 2
        noise = np.random.normal(0, sigma, mat.shape)
        res = mat + noise * 100 #(135.0 - np.abs(mat-120.0))
        res = cv2.GaussianBlur(res, (5,5), 2)
        mat[:] = np.clip(res, 0, 255)


class VisionSensorNoiseModel(object):
    GAUSSIAN    = VisionSensorNoiseGaussian
    SALT_PEPPER = VisionSensorNoiseSaltPepper
    SPECKLE     = VisionSensorNoiseSpeckle





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
        start = np.array(location).astype(np.int)
        end = start + np.array(size).astype(np.int)
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
        pos = tuple(location.astype(np.int))
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
        return cv2.matchTemplate(observation, world, cv2.TM_CCOEFF)

    @classmethod
    def evaluate_at(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        # norm = np.sqrt(np.sum(model*model) * np.sum(observation*observation))
        # return np.sum( (model - np.mean(model)) * (observation - np.mean(observation)) ) / norm
        return np.sum( (model - np.mean(model)) * (observation - np.mean(observation)) )


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
