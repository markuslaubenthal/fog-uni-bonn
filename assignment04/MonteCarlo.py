# Authors: Markus Laubenthal, Bilal Kizilkaya, Lennard Alms


import numpy as np
import matplotlib.pyplot as plt
from math import pi
from numpy import sin, cos # we need the vectorized versions from numpy!
# Part a)


# number of random chosen points
limit = 1000
plotAt = limit/10 # the iteration at which we want to see the plot

radius = 3
area_circle = pi*radius**2 # used to compute the error later on
area_approx = np.zeros((limit))

for N_points in range(1, limit):
	if(0 == N_points % 10):
		print(N_points, "/", limit)

	# randomly select N_points 2D points in the enclosing rectangle
	point_coords = np.random.rand(2, N_points) * 6 - 3

	assert(point_coords.shape == (2, N_points))

	# calculate the radii for all of the points
	#Rsquare = ...

	# check which points are inside
	check = ((point_coords[0] - 0)**2 + (point_coords[1] - 0)**2) < radius**2
	assert(len(check) == N_points)

	# count inliers
	inside_counter = np.sum(check)

	# approximate area
	area_approx[N_points] =  (4 * inside_counter / N_points) * radius ** 2

	# show a plot
	if (N_points==plotAt):
		# find inlier/outlier
		[inlier_idx] = np.nonzero(check)
		[outlier_idx] = np.nonzero(~check)
		# plot inlier/outlier
		plt.scatter(point_coords[0,inlier_idx], point_coords[1,inlier_idx], color='blue')
		plt.scatter(point_coords[0,outlier_idx], point_coords[1,outlier_idx], color='green')
		phi = np.arange(0, 2*pi, 0.01)
		x_c = radius*np.cos(phi)
		y_c = radius*np.sin(phi)
		plt.plot(x_c, y_c, color='b')
		plt.axis('equal')
		plt.show(block=False)



# calculate the error for different numbers of points
error = np.absolute(area_circle - area_approx)
print("Calculated Area of Circle: ", area_approx[-1])
assert(len(error) == limit)

# plot the error
plt.figure()
plt.title("Errors Circle")
plt.plot(range(len(error)), error)
plt.show(block=False)


##############################################################
##



from mpl_toolkits.mplot3d import Axes3D

# Part b)

# number of random chosen points
limit = 1000
plotAt = limit/10 # the iteration at which we want to see the plot

radius = 3
volume_sphere = 4/3*pi*radius**3 # used to compute the error later on
volume_approx = np.zeros((limit))

for N_points in range(1, limit):
	if(0 == N_points % 10):
		print(N_points, "/", limit)

	# randomly select N_points 3D points in the enclosing cube
	point_coords = np.random.rand(3, N_points) * 6 - 3

	assert(point_coords.shape == (3, N_points))

	# calculate the radii for all of the points
	#Rsquare = ...

	# check which points are inside
	check = ((point_coords[0])**2 + (point_coords[1])**2 + (point_coords[2])**2) < radius**2

	assert(len(check) == N_points)

	# count inliers
	inside_counter = np.sum(check)

	# approximate area
	# inside_counter / N_points = 1/6 * pi
	volume_approx[N_points] =  4/3 * (6 * inside_counter / N_points) * radius ** 3


	# show a plot
	if (N_points==plotAt):
		# find inlier/outlier
		[inlier_idx] = np.nonzero(check)
		[outlier_idx] = np.nonzero(~check)
		# plot inlier/outlier
		plt.figure()
		ax = plt.gcf().add_subplot(111, projection='3d')
		ax.scatter(point_coords[0,inlier_idx], point_coords[1,inlier_idx], point_coords[2,inlier_idx], color='blue')
		ax.scatter(point_coords[0,outlier_idx], point_coords[1,outlier_idx], point_coords[2,outlier_idx], color='green')
		plt.show(block=False)



# calculate the error for different numbers of points
error = np.absolute(volume_sphere - volume_approx)
print("Calculated Volume of Sphere: ", volume_approx[-1])
assert(len(error) == limit)

# plot the error
plt.figure()
plt.title("Errors Sphere")
plt.plot(range(len(error)), error)
plt.show()
