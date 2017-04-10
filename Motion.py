
import pygame
import numpy as np
import scipy.stats as stats

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
    def add(motion, level_pos, level_rot):
        # by distance and rotation
        sigma_pos = level_pos
        noise_pos = stats.norm.rvs(0, sigma_pos, 2)
        sigma_rot = level_rot
        noise_rot = stats.norm.rvs(0, sigma_rot, 1)
        motion += np.hstack( (noise_pos, noise_rot) )

    @staticmethod
    def prob(motion, motion_ref, level_pos, level_rot):
        # by distance and rotation
        diff = (motion - motion_ref).T
        sigma_pos = level_pos
        prob = stats.norm.pdf(diff[0,], 0, sigma_pos)
        sigma_rot = level_rot
        noise_rot = stats.norm.pdf(diff[2,], 0, sigma_rot)


class MotionSensorNoiseModel(object):
    GAUSSIAN  = MotionSensorNoiseGaussian






class Motion(object):

    @classmethod
    def propagate(cls, motion, locations=None):
        pos = -np.array(observation.get_size())/2
        size = world.size + observation.get_size() - 1
        model = world.map(np.concatenate((pos, size)))
        model_m = pygame.surfarray.pixels2d(model)
        observation_m = pygame.surfarray.pixels2d(observation)
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

