
import pygame
import numpy as np
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

import cv2

import noise

from matplotlib import cm
from matplotlib.colors import Colormap

mapping = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.45, 0.70, 0.90, 0.95, 0.97, 0.99, 1.00])

def world_type_from_colormap(cmap, mapping = mapping):
    x = np.linspace(0,1,len(mapping))
    res = np.roll(cmap(x), 1, -1)
    res[:,0] = mapping
    return res

def create_palette(colors):
    if colors is None:
        c = np.linspace(0, 1, 256)
        cm = np.vstack([c, c, c]).T
    elif isinstance(colors, Colormap):
        cm = colors(np.linspace(0, 1, 256))
    else:
        c = interpolate.interp1d(colors[:,0], colors[:,1:], axis=0)
        cm = c(np.linspace(0, 1, 256))

    return (cm*255).astype(np.int32)



class WORLD_TYPE(object):

    @staticmethod
    def get(name):
        if name not in WORLD_TYPE.__dict__: return None
        return WORLD_TYPE.__dict__[name]

    NONE = np.array([[0.0,  0.3, 0.6, 0.2], [1.0,  0.3, 0.6, 0.2]])

    GREY_RAW    = cm.gray
    TERRAIN_RAW = cm.terrain
    OCEAN_RAW   = cm.ocean
    EARTH_RAW   = cm.gist_earth
    MARS_RAW    = cm.Set1

    GREY    = world_type_from_colormap(cm.gray)
    TERRAIN = world_type_from_colormap(cm.terrain)
    OCEAN   = world_type_from_colormap(cm.ocean)
    EARTH   = world_type_from_colormap(cm.gist_earth)
    MARS    = world_type_from_colormap(cm.Set1)

    MY_EARTH = np.array([
        [0.00,  0.00, 0.00, 0.40],    # base
        [0.05,  0.00, 0.00, 0.70],    # water
        [0.10,  0.20, 0.40, 0.80],    # shallow water
        [0.15,  0.70, 0.65, 0.45],    # beach
        [0.20,  0.10, 0.50, 0.10],    # bushes
        # [0.45,  0.40, 0.90, 0.20],    # grass
        [0.45,  0.30, 0.70, 0.20],    # grass
        # [0.50,  0.50, 0.60, 0.20],    # savanna
        [0.70,  0.40, 0.90, 0.20],    # grass
        [0.90,  0.00, 0.50, 0.10],    # forest
        [0.95,  0.40, 0.70, 0.20],    # grass
        [0.97,  0.50, 0.50, 0.50],    # rock
        [0.99,  0.40, 0.40, 0.40],    # rock
        [1.00,  1.00, 1.00, 1.00]     # snow
    ])

    MY_MARS = np.array([
        [0.00,  0.20, 0.00, 0.00],    # base
        [0.05,  0.40, 0.20, 0.20],    # water
        # [0.10,  0.50, 0.30, 0.10],    # shallow water
        [0.35,  0.80, 0.50, 0.20],    # shallow water
        [0.60,  0.60, 0.40, 0.30],    # shallow water
        [0.90,  0.90, 0.60, 0.50],    # snow
        [0.95,  0.90, 0.60, 0.50],    # snow
        [1.00,  0.90, 0.70, 0.60]     # snow
    ])



class Generator(object):

    def __init__(self, width, height, scale_x = 6, scale_y = 6, octaves = 6, seed = 0):
        self.width = width
        self.height = height
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.octaves = octaves
        self.seed = seed

    @property
    def size(self): return np.array([self.width, self.height])

    @size.setter
    def size(self, size): return np.array([size[0], size[1]])

    def _eval_at(self, x, y):
        raise NotImplementedError("Use a specialized generator")

    def at(self, points):
        points = points.astype(np.float)
        points[:,0] *= self.scale_x / float(self.width)
        points[:,1] *= self.scale_y / float(self.height)

        res = np.array([self._eval_at(x, y) for x,y in points])
        res = (res + 1)/2.0
        return res


    def create(self, area, width, height):
        Xs = np.linspace(area.left, area.right, width+1)
        Ys = np.linspace(area.top, area.bottom, height+1)

        Y, X = np.meshgrid(Ys, Xs)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        return np.array(self.at(points)).reshape([width+1, height+1])[0:-1,0:-1]


class GeneratorPerlin(Generator):
    def _eval_at(self, x, y):
        return noise.pnoise2(x, y, self.octaves, repeatx=self.scale_x, repeaty=self.scale_y, base=self.seed, persistence=0.4, lacunarity=3.3)

class GeneratorSimplex(Generator):
    def _eval_at(self, x, y):
        return noise.snoise2(x, y, self.octaves, repeatx=self.scale_x, repeaty=self.scale_y, base=self.seed, persistence=0.4, lacunarity=3.3)


class NOISE(object):
    PERLIN = 0
    SIMPLEX = 1


class MapGenerator(object):

    def __init__(self, width, height, scale = 6, seed = 0, colors = None, type = None):
        self.width = width
        self.height = height
        self.seed = seed
        self.colors = colors

        if type == NOISE.PERLIN:
            self._gen = GeneratorPerlin(width, height, scale, scale, 6, seed)
        elif type == NOISE.SIMPLEX:
            self._gen = GeneratorSimplex(width, height, scale, scale, 6, seed)
        else:
            self._gen = GeneratorSimplex(width, height, scale, scale, 6, seed)


    @property
    def size(self): return np.array([self.width, self.height])

    def set_colors(self, colors):
        self.colors = colors

    def create(self, area=None, width=None, height=None):
        if area is None: area = pygame.Rect( [0,0], self.size)
        if width is None: width = self.width
        if height is None: height = self.height

        genmap = pygame.Surface([width, height], 0, 8)

        hmap = self._gen.create(area, width, height)
        hmap = (hmap*255).astype(np.uint8)
        cv2.equalizeHist( hmap, hmap );

        palette = create_palette(self.colors)
        genmap.set_palette(palette)

        pygame.surfarray.blit_array(genmap, hmap)

        return genmap



class ExtendedMapGenerator(MapGenerator):

    def __init__(self, width, height, pad_x = 0, pad_y = 0, scale = 6, seed = 0, colors = None, type = None):
        super(ExtendedMapGenerator, self).__init__(width, height, scale, seed, colors, type)
        self.pad_x = pad_x
        self.pad_y = pad_y

    @property
    def pad(self):
        return np.array([self.pad_x, self.pad_y])

    @property
    def width_ext(self):
        return self.width + 2*self.pad_x

    @property
    def height_ext(self):
        return self.height + 2*self.pad_y

    @property
    def size_ext(self):
        return self.size + 2*self.pad

    def create(self, area=None, width=None, height=None):
        if area is None: area = pygame.Rect(-self.pad, self.size_ext)
        if width is None: width = self.width_ext
        if height is None: height = self.height_ext
        return super(ExtendedMapGenerator, self).create(area, width, height)



class WorldMap(object):

    def __init__(self, width, height, pad_x = 0, pad_y = 0, scale = 6, seed = 0, colors = None, type = None):
        self._mapper = ExtendedMapGenerator(width, height, pad_x, pad_y, scale, seed, colors, type)
        self._map = self._mapper.create()

    @property
    def width(self): return self._mapper.width

    @property
    def height(self): return self._mapper.height

    @property
    def size(self): return self._mapper.size

    def map(self, roi = None, size = None):
        if roi is None: roi = [0, 0, self._mapper.width, self._mapper.height]
        roi[0] += self._mapper.pad_x
        roi[1] += self._mapper.pad_y

        if size is None:
            rect = pygame.Rect( (roi[0], roi[1]), (roi[2], roi[3]) )
            submap = self._map.subsurface(rect)
            submap = submap.copy()
        else:
            rect = pygame.Rect( (roi[0]-1, roi[1]-1), (roi[2]+2, roi[3]+2) )
            submap = self._map.subsurface(rect)
            scale_x = float(size[0]) / roi[2]
            scale_y = float(size[1]) / roi[3]
            submap = pygame.transform.scale(submap, [int(size[0] + 2*scale_x), int(size[1] + 2*scale_y)])
            subpix_x, _ = np.modf(roi[0])
            subpix_y, _ = np.modf(roi[1])
            dx = int(np.round(-subpix_x * scale_x))
            dy = int(np.round(-subpix_y * scale_y))
            submap.scroll(dx, dy)
            view_m = pygame.surfarray.pixels2d(submap)
            gaussian_filter(view_m, sigma=(scale_x/3.0, scale_y/3.0), output=view_m)
            del view_m


        return submap
