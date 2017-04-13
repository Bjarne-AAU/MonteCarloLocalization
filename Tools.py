
import numpy as np
import pygame

def as_int(f):
    return np.rint(f).astype(np.int)

class Timer(object):

    def __init__(self, framerate=None):
        self._last = pygame.time.get_ticks()
        self._dt = None if framerate is None else 1000.0/framerate

    def tic(self):
        current = pygame.time.get_ticks()
        passed = current - self._last
        has_passed = self._dt is None or passed > self._dt
        if has_passed: self._last = current
        return has_passed

    def toc(self):
        current = pygame.time.get_ticks()
        return current - self._last
