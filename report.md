# MonteCarloLocalization

Some bla bla goes here


## Derivation of sequential Bayes filter

Some definitions and mathematical derivation of the Bayes filter

## Sensor models and noise

Sensors can be described by their observation method and their observation rate.
The observations made by the sensor underly an unknown noise model.
The abstract sensor and noise model are implemented in the file "Sensor.py"

```
class Sensor(object):
    def __init__(self, fps=None):...
    def set_noise_model(self, noise): ...
    def observation(self): ...
    def observe(self, what): ...
```

In our case, we simulate two kinds of sensors with different noise models:

 * Vision sensor: Captures height of terrain beneath the spacecraft
 * Motion sensor: Captures motion of the spacecraft (Odometry/Position)

### Vision Sensor

The vision sensor is able to capture the height of the terrain in its view.
The view is determined by the position of the spacecraft and is able to get information about the terrain within a certain range given by a rectangle with a predefined size.
The sensor is capable to make observations at a given framerate.
The implementation of the Vision sensor can be found in the file "Vision.py".


Additionally, we implemented various methods for simulating sensor noise:

 * Gaussian
 * Salt
 * Pepper
 * Salt & Pepper
 * Speckle


### Motion Sensor

The implementation of the Motion sensor can be found in the file "Motion.py".
We differentiate between odometry (rotation, distance) and vector measurements.

This is reflected by the two noise models:

 * Gaussian (vector)
 * Advanced (rotation, distance)




## Observation model



## Motion model

## Respresentations
