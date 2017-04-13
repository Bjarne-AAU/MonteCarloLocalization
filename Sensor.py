
from Tools import Timer

class Sensor(object):

    def __init__(self, fps=None):
        self._timer = Timer(fps)
        self._observation = None
        self._observation_noisy = None
        self._noise = None

    def set_noise_model(self, noise):
        self._noise = noise

    @property
    def observation(self):
        return self._observation_noisy

    def observe(self, what):
        if not self._timer.tic(): return False
        self._observation = self._observe(what)
        if self._noise is not None:
            self._observation_noisy = self._noise.add(self._observation)
        else:
            self._observation_noisy = self._observation

        return True

    def draw(self, image):
        if self._observation_noisy is not None:
            self._draw(image)

    def _observe(self, what):
        raise NotImplementedError()

    def _draw(self, image):
        raise NotImplementedError()


class SensorNoise(object):

    def __init__(self, level):
        self.level = level

    def create(self, observation):
        raise NotImplementedError()

    def add(self, observation):
        return observation + self.create(observation)
