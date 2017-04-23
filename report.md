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


## Motion model



In the observation model it is calculated the likelihood of being in the center of a map patch $M$ based on the observation $Z$ from the vision sensor. This likelihood can be evaluated using four different methods:

 * Mean absolute difference. It is more suited for the particle-based method as it introduces more noise to the comparison. The mean produces a flatter distribution, which helps to avoid that positions with high similarities are not covered by particles.

 	$ R(x,y)= \sum_{x',y'} |Z(x',y')-M(x+x',y+y')| $

    Implementation:
    ```
    def ObservationMDIFF(cls, location, world, observation):
        pos = tuple(location)
        observation_mean = np.mean(observation)
        return -np.abs(world[pos] - observation_mean)
    ```

 * Normalized cross-correlation coefficient. It is better suited for the grid-based method as it finds positions with high similarity.

    $ \begin{array}{l} Z'(x',y')=Z(x',y') - 1/(w \cdot h) \cdot \sum _{x'',y''} Z(x'',y'') \\ M'(x+x',y+y')=M(x+x',y+y') - 1/(w \cdot h) \cdot \sum _{x'',y''} M(x+x'',y+y'') \end{array} $

 	$ R(x,y)= \frac{\sum_{x',y'} (Z(x',y') \cdot M(x+x',y+y'))}{\sqrt{\sum_{x',y'}Z(x',y')^2 \cdot \sum_{x',y'} M(x+x',y+y')^2}} $

    Implementation:
    ```
    def ObservationCCOEFF(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        norm = np.sqrt(np.sum(model*model) * np.sum(observation*observation))
        return np.sum( (model - np.mean(model)) * (observation - np.mean(observation)) ) / norm
    ```
 * Normalized cross-correlation. Similar to the previous method. The main difference is that is not normalized, so it will perform worse when dealing with large values.

 	$ R(x,y)= \frac{ \sum_{x',y'} (Z'(x',y') \cdot M'(x+x',y+y')) }{ \sqrt{\sum_{x',y'}Z'(x',y')^2 \cdot \sum_{x',y'} M'(x+x',y+y')^2} } $

    Implementation:
    ```
    def ObservationCCORR(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        norm = np.sqrt(np.sum(model*model) * np.sum(observation*observation))
        r = np.sum( model * observation ) / norm
        return ((r**2)**2)**2
    ```

 * Mean squared difference. It is a good compromise in performance for the grid-based and particle-based method.

 	$ R(x,y)= \sum_{x',y'} (Z(x',y')-M(x+x',y+y'))^2 $

    Implementation:
    ```
    def ObservationSQDIFF(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        diff = observation - model
        return -np.sum(diff*diff)
    ```

## Respresentations
