---
title: Monte Carlo Localization
author:
- Andreas Munk
- Jose Armenteros
- Ramido Mata
- Bjarne Grossmann
---



# Introduction

Monte Carlo localization is a method common in mobile robotics that enables a robot to estimate its location within a known environment using a recursive Bayesian estimation approach.
The robot is equipped with different sensors to sense and observe the environment as it moves around.
Using a given map, it then uses the observations to sequentically infer its location within that map.

In Bayesian terms, the recursive Bayesian estimation (or Bayesian filter) is a general method to estimate an unknown probability density function over time by incorporating new observation through an observation and a processing model.
The probability density function represents the belief over the states of a dynamic system and can therefore be seen a a filtering process to eliminate less probable states and eventually estimate the true underlying state.
The computation of the posterior distribution is often untractable due to the complexity of the observation model and the dynamically changing state and is therefore often approximated by grid-based or particle-based Monte Carlo methods.

In this project, we developed an demonstration application that shows the mechanism behind the recursive Bayesian estimation for grid- and particle-based Monte Carlo methods in a localization scenario.
In this simulation, we send a autonomous robot or spacecraft to a remote area or planet which topological map is known beforehand.
The spacecraft is equipped with a vision sensor observing the height of the terrain nearby, and a motion sensor measuring the motion of the spacecraft.
Using these noisy observations, the task is to estimate our location with respect to the given topological map.


# Problem statement

The goal of the Monte Carlo localization is to estimate the state of the spacecraft - in this particular case, we are interested only in the location $(x,y)$ on the topological map $\mathcal{M}$ with size $(x_{max}, y_{max})$.
The state space $\Omega$ can be represented by all possible states
$$ \theta = (x,y) \in \Omega \quad \text{ with } 0 < x < x_{max} \text{ and } 0 < y < y_{max} $$
and our goal is to find a good estimate $\hat{\theta_t}$ of the real state $\theta_t$ at time t.
We can formulate the belief $Bel(\theta_t)$ of possible states at a time $t$ given the independent observations $z_{0:t}$ and motions $u_{0:t}$ (note that we do not explicitly state the conditioning on $u_{0:t}$ for readability) in a probabilistic manner as
$$ Bel(\theta_{t}) = p(\theta_t | z_{0:t})
                   = \frac{p(z_t | \theta_t) \ p(\theta_t | z_{0:t-1})}{p(z_t | z_{0:t-1})} $$

with the normalization term
$$ p(z_t | z_{0:t-1}) = \int p(z_t | \theta_t) \ p(\theta_t | z_{0:t-1}) d\theta_t $$
and the predictive distribution
$$ p(\theta_t | z_{0:t-1}) = \int p(\theta_t | \theta_{t-1}) \ p(\theta_{t-1} | z_{0:t-1} ) \quad.$$

The system can be fullydetermined by defining the **observation model**
$p(z_t | \theta_t) = p(z_t | \theta_t, \mathcal{M})$
to compute the likelihood of the observation $z_t$ at location $\theta_t$ given the map model $\mathcal{M}$, and the **motion model** given by the transition distribution
$p(\theta_t | \theta_{t-1}) = p(\theta_t | \theta_{t-1}, u_t)$
given the last estimated location $\theta_{t-1}$ and the observed motion $u_t$.

However, the integrals of the normalization term and the predictive distribution cannot be determined analytically and have to be solved using the Monte Carlo integration method.
With the Monte Carlo representation of the posterior distribution, we are able to estimate the true state of the spacecraft with respect to any criterion.
Typical estimates are the Minimum Mean Square Estimate (or the expectated value)
$$ \hat{\theta}_t^{\scriptscriptstyle{\text{MMS}}} = E(\theta_t | z_{0:t}) = \int \theta_t p(\theta_t | z_{0:t}) d\theta_t $$
or the Maximum A-Posteriori estimate (or the maximum mode)
$$ \hat{\theta}_t^{\scriptscriptstyle{\text{MAP}}} = \text{arg}\max_{\theta_t}(\theta_t | z_{0:t}) $$


# Derivation of sequential Bayes filter

Some definitions and mathematical derivation of the Bayes filter


# Simulation

In order to demonstrate the Monte Carlo localizatin method, a model of a real scenario has to be created in which the simulation takes place.
Therefore, we need to model

 * the environment given as a topolgical map
 * the spacecraft which is able to navigate around the map
 * the sensors used by the spacecraft to make observations


## World model

The world model is generated automatically and is represented by a height map, i.e. a rasterized image is used to represent the elevation at discrete positions.
The height values are created using a composition of perlin or simplex noise which is often used in computer graphics to create natural landscapes.
The height values are then mapped to colors for visual purpose only.
The world map is implemented in the file [World.py](http://github.com/Bjarne-AAU/MonteCarloLocalization/blob/master/World.py)

![A typical map generated by our algorithm.](figures/world_model.png)


## Spacecraft model

The spacecraft is basically just a location which can be moved around manually using the arrow keys.
The motion is modeled using an odometric model, i.e. the spacecraft's movement is determined by it's orientation and speed.
The spacecraft is implemented in the file [Robot.py](http://github.com/Bjarne-AAU/MonteCarloLocalization/blob/master/Robot.py)

![The spacecraft.](figures/rocket.png)


## Sensor model

Sensors can be described by their observation method and their observation rate.
The observations made by the sensor underly an unknown noise model.
The abstract sensor and noise model are implemented in the file [Sensor.py](http://github.com/Bjarne-AAU/MonteCarloLocalization/blob/master/Sensor.py).

In our case, we simulate two kinds of sensors with different noise models:

 * Vision sensor: Captures height of terrain beneath the spacecraft
 * Motion sensor: Captures motion of the spacecraft (Odometry/Position)

### Vision Sensor

The vision sensor is able to capture the height of the terrain in its view.
The view is determined by the position of the spacecraft and is able to get information about the terrain within a certain range given by a rectangle with a predefined size.
The sensor is capable to make observations at a given framerate.
The implementation of the Vision sensor can be found in the file [Vision.py](http://github.com/Bjarne-AAU/MonteCarloLocalization/blob/master/Vision.py).

Additionally, we implemented various methods for simulating sensor noise:

 * Gaussian: Simple pixelwise Gaussian noise
 * Salt: Some pixels are changed to the maximum possible value
 * Pepper: Some pixels are changed to the minimum possible value
 * Salt & Pepper: Some pixels are changed randomly to the maximum or minimum possible value
 * Speckle: Additive Gaussian noise based on observed pixels

![Different noise models for the vision sensor.](figures/vision_noise.png)

The implementation of the Vision sensor noise can also be found in the file [Vision.py](http://github.com/Bjarne-AAU/MonteCarloLocalization/blob/master/Vision.py).

### Motion Sensor

The motion sensor is used to capture the movement of the spacecraft since the last measurement and is basically just the motion vector between the last and current position.
The sensor is capable to make observations at a given framerate.
The implementation of the Motion sensor can be found in the file [Motion.py](http://github.com/Bjarne-AAU/MonteCarloLocalization/blob/master/Motion.py).

We simulate two different measurement models: (a) using the absolute position measurements or (b) using the odometry, i.e. independent measurement of the angle and distance of the motion.
This is reflected by the two noise models:

 * Position: The positional error is given by a bivariate normal distribution
 * Odometry: Rotation and distance are both normal distributed, but considered independently

![Error distribution of different noise models for the motion sensor. The position after the true motion would be in the center of the image. ](figures/motion_noise.png)



# Motion model



# Observation model

In the observation model it is calculated the likelihood of being in the center of a map patch $M$ based on the observation $Z$ from the vision sensor. This likelihood can be evaluated using four different methods:

 * Mean absolute difference. It is more suited for the particle-based method as it introduces more noise to the comparison. The mean produces a flatter distribution, which helps to avoid that positions with high similarities are not covered by particles.

 	$$ R(x,y)= \sum_{x',y'} |Z(x',y')-M(x+x',y+y')| $$

    Implementation:
    ```python
    def ObservationMDIFF(cls, location, world, observation):
        pos = tuple(location)
        observation_mean = np.mean(observation)
        return -np.abs(world[pos] - observation_mean)
    ```

 * Normalized cross-correlation coefficient. It is better suited for the grid-based method as it finds positions with high similarity.

    $$ \begin{array}{l} Z'(x',y')=Z(x',y') - 1/(w \cdot h) \cdot \sum _{x'',y''} Z(x'',y'') \\ M'(x+x',y+y')=M(x+x',y+y') - 1/(w \cdot h) \cdot \sum _{x'',y''} M(x+x'',y+y'') \end{array} $$

 	$$ R(x,y)= \frac{\sum_{x',y'} (Z(x',y') \cdot M(x+x',y+y'))}{\sqrt{\sum_{x',y'}Z(x',y')^2 \cdot \sum_{x',y'} M(x+x',y+y')^2}} $$

    Implementation:
    ```python
    def ObservationCCOEFF(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        norm = np.sqrt(np.sum(model*model) * np.sum(observation*observation))
        return np.sum( (model - np.mean(model)) * (observation - np.mean(observation)) ) / norm
    ```
 * Normalized cross-correlation. Similar to the previous method. The main difference is that is not normalized, so it will perform worse when dealing with large values.

 	$$ R(x,y)= \frac{ \sum_{x',y'} (Z'(x',y') \cdot M'(x+x',y+y')) }{ \sqrt{\sum_{x',y'}Z'(x',y')^2 \cdot \sum_{x',y'} M'(x+x',y+y')^2} } $$

    Implementation:
    ```python
    def ObservationCCORR(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        norm = np.sqrt(np.sum(model*model) * np.sum(observation*observation))
        return np.sum( model * observation ) / norm
    ```

 * Mean squared difference. It is a good compromise in performance for the grid-based and particle-based method.

 	$$ R(x,y)= \sum_{x',y'} (Z(x',y')-M(x+x',y+y'))^2 $$

    Implementation:
    ```python
    def ObservationSQDIFF(cls, location, world, observation):
        model = cls._extract_location(location, observation.shape, world)
        diff = observation - model
        return -np.sum(diff*diff)
    ```

# Respresentations

We approximate the probability distribution (eq. XX) using two methods: grid-based method and particle filter. In grid-based localization (assuming the robot is stationary) the probabilities of all the coordinates in the grid are evaluated. Thus, the robot's belief of its location in the environment is represented by those probabilities. Because all the grid coordinates are evaluated, the grid-based method has less tractable computational properties. That is, the memory requirements and computational time scales with the size and resolution of the environment considered. In our case, we represent the environment as a 2-dimensional x,y coordinate map plus the altitude (i.e. distance between robot and map's surface).

With the particle-filter, the robot's belief is evaluated from randomized weighted samples of its environment (i.e. particles). Therefore, the computational load is more tractable compared to the grid-based method as it scales linearly with the amount of particles employed. However, this comes with the trade-off of less accurate approximations compared to the grid-based method.

PROPAGATING DENSITIES PART GOES HERE: In essence, the difference between these two methods during the recursive steps can be thought of as in one the probability densities are propagated from the previous state to the current state (grid-based), whereas in the other the pdf is ... (I don't know what I'm talking about...lol)



## Resampling Methods

We implement three different sampling methods for the robot localization problem: naive sampling, stratified sampling, and systematic sampling. Ideally, the goal of the resampling method should be to sample those particles that are representative of our unknown probability distribution (eq. XX, the probability of the robot's location). In other words, to filter out unlikely particles and choose those with a highr probability (higher weights). However, sampling randomly can result in systematically picking more low weight particles and discarding large weight particles, and can therefore not only result in an unrepresentative sample, but it can also gradually lead to degeneracy as the sampling sequence continues. We employ these resampling methods to combat degeneracy (i.e. to not discard highly weighted particles during the resampling phase). We briefly expose the three methods and their properties. All three methods share the same initial structure in the sense that we first compute a normalized cumulative sum of the weights and then we sample with replacement from it. The following function implements the normalized cumulative sum and the draws from it with the index supplied by the implemented resampling method.

```python
def _resample_indices(self, positions):
        indices = np.searchsorted(np.cumsum(self._density), positions, side='right')
        self._locations = self._locations[indices]
        self._density = self._density[indices]
        self._density /= np.sum(self._density)
```

### Naive Sampling

In naive sampling, we simply draw uniformly from this cumulative sum of the weights. We implement this in python by cross-referencing two arrays: one array which stores N uniform draws, and another storing the cumulative sum of the weights (see function above). We then use the index of the drawn particles from the cumulative sum array as the basis of our new sample distribution. In this manner, particles with larger weights are more likely to be drawn, and hence we prevent the problem of picking unrepresentative particles.

```python
def resample_naive(self):
        positions = np.random.random(self.N)
        self._resample_indices(positions)
```

### Stratified Sampling

Similar to naive sampling, we compute the cumulative sum of the weights and normalize it. However, it differs from it in that we divide our sampling space into N equal groups (i.e. the stratification), then we draw uniformly from each group. This results in drawing the same amount of times from each subdivision, and hence we ensure that we sample more evenly across the sampling space.

```python
def resample_stratified(self):
        positions = (np.arange(self.N) + np.random.random(self.N)) / self.N
        self._resample_indices(positions)
```

However, this introduces a new problem - sample impoverishment - whereby with each iteration our sample becomes more and more highly concentrated with particles originating from large weight particles. That is, since large weight particles are more likely to be drawn, it can be seen that with each resampling step, we gradually reduce the diversity of the particles. In a worst case scenario (in the limit) of sample impoverishement, we would end up with a sample whose particles originated from one particle of large weight, which would result in (poorly) approximating our probability distribution (eq. XX) with only one point estimate. We combat this by resampling only when the effective sample size (N_eff is set to 0.5 here) threshold is reached.  In this manner, we slow down the process of sample impoverishement.


### Systematic Sampling

Similar to stratified sampling, we divide the sampling space into N equal groups, except that instead of each group's draw being independent of eachother, in systematic sampling the position of a draw is the same in each group. Therefore, the draws in each group are equally spaced.

```python
def resample_systematic(self):
        positions = (np.arange(self.N) + np.random.random()) / self.N
        self._resample_indices(positions)
```
