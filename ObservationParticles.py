
import numpy as np
from scipy.spatial import distance 
import pygame

def weights_likelihood(particles,robot_size, area_m,world):
		particle_m = pygame.surfarray.array2d(world.map(np.concatenate((particles,robot_size))))

		#new_weights = distance.cdist(particle_m, area_m, 'sqeuclidean')
		#print new_weights.shape


		return -np.linalg.norm(particle_m - area_m)

def evaluate_particles_weights(robot,particles,world):
        # Observation
        area = world.map(list(robot.pos) + list(robot.size))

        N = particles.N

        robot_size_tmp = np.repeat(robot.size,N,axis=0)

        # Particle map shape (N,size_x,size_y)
		#particles_area = np.apply_along_axis(world.map, axis=0, np.concatenate((particles.positions,robot_size_tmp)))      


        #map_m = pygame.surfarray.array2d(particles_area).astype(np.float)
        area_m = pygame.surfarray.array2d(area).astype(np.float)

        randn = np.random.normal(0, 5.0, area_m.shape)# * 30 / scale
        area_m += randn

        area_m[area_m > 255.0] = 255.0
        area_m[area_m < 0.0] = 0.0

        new_weights = np.apply_along_axis(weights_likelihood, 1, particles.positions, robot_size_tmp, area_m, world)
        if (np.max(new_weights) > np.min(new_weights)):
        	new_weights = (new_weights - np.min(new_weights))/(np.max(new_weights)-np.min(new_weights))
        #posterior_weights = particles.weights * new_weights 
        else: new_weights[:] = 1

        return new_weights



	