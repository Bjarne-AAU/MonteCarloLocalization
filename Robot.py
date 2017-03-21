import pygame
import numpy as np

class Robot(object):

    def __init__(self, x, y, angle, width, height, filename):
        self.x = x
        self.y = y
        self.angle = angle

        self._v = 0.0
        self._r = 0.0

        self.set_image(filename)
        self._width = width
        self._height = height
        self.scale = 1.0
        self._color = (0,0,0)
        self._alpha = 255

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def width(self):
        return int(self.scale * self._width + 0.5)

    @property
    def height(self):
        return int(self.scale * self._height + 0.5)

    @property
    def size(self):
        return np.array([self.width, self.height])

    @property
    def color(self):
        return self._color + (self._alpha,)

    def set_color(self, r, g, b, a = None):
        self._color = (r,g,b)
        if a is not None: self.set_alpha(a)

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_image(self, filename):
        self._image = pygame.image.load(filename).convert()

    @property
    def image(self):
        image = pygame.Surface(self._image.get_size(), pygame.SRCALPHA)

        center1 = np.array(self._image.get_rect().center)
        rot = pygame.transform.rotate(self._image, self.angle)
        center2 = np.array(rot.get_rect().center)
        rot.set_colorkey(rot.get_at((0,0)))

        image.blit(rot, (center1-center2))
        image.fill(self.color, special_flags=pygame.BLEND_RGBA_MULT)
        return pygame.transform.smoothscale(image, self.size)

    def accelerate(self, vx, dt):
        self._v += vx * dt

    def rotate(self, vr, dt):
        self._r += vr * dt

    def update(self, dt):

        self.x += np.sin(self.angle/180.0*np.pi) * self._v * dt
        self.y += np.cos(self.angle/180.0*np.pi) * self._v * dt
        self.angle += self._r * dt

        self._v -= (1.0*self._v) * dt
        self._r -= (2.5*self._r) * dt
