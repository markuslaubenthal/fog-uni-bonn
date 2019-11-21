# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:59:44 2015

@author: merzbach
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage import feature, io
from skimage.filters import sobel
from math import pi, log, sqrt


# --------------- Framework functions (nothing to do here!) ---------------

# displays a set of circles, specified by three arrays containing x- & y-coordinates, as well as circle radius; optionally a score can be specified that is visulalized via the circle color (brighter color == higher score)
def DrawCircle(axes_handle, centers_x, centers_y, radii):
	n = 100 # number of samples for drawing circles
	# iterate over all circles that are passed
	for x, y, radius in zip(centers_x, centers_y, radii):
		angles = np.arange(0, (2. - 2. / n) * pi, 1. / n)
		(xs, ys) = (radius * np.cos(angles) + x, radius * np.sin(angles) + y)
		
		axes_handle.plot(xs, ys, color="C2", linewidth=3)
	
	
# function for fitting a circle to a set of points
# returns (center, radius)
def FitCircle(sample_points, axes_handle=None):
	N = sample_points.shape[1]
	x_avg = np.mean(sample_points[0, :])
	y_avg = np.mean(sample_points[1, :])
	u = sample_points[0, :] - x_avg
	v = sample_points[1, :] - y_avg
	
	# See http://www.dtcenter.org/met/users/docs/write_ups/circle_fit.pdf for a derivation of the following.
	Suu = u.dot(u)
	Svv = v.dot(v)
	Suv = u.dot(v)
	Suuu = np.sum(u * u * u)
	Svvv = np.sum(v * v * v)
	Suvv = np.sum(u * v * v)
	Svuu = np.sum(v * u * u)

	A = np.array([[Suu, Suv], [Suv, Svv]])
	b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
	try:
		center = np.linalg.solve(A, b)
	except np.linalg.LinAlgError:
		return ([-1, -1], -1)
	
	radius = sqrt(center.dot(center) + (Suu + Svv) / N)
	center = center + np.array([x_avg, y_avg])
	
	if not axes_handle is None:
		DrawCircle(axes_handle, [center[1]], [center[0]], [radius])
		axes_handle.scatter(sample_points[1, :], sample_points[0, :], s=50, c='r', linewidths=0)
		axes_handle.scatter(center[1], center[0])
	return (center, radius)



def LoadImage(filename):
	# read image & compute edge map
	img = io.imread(filename)
	
	# convert to gray image if necessary
	if len(img.shape) > 2:
		img = img[:, :, 0]
	
	return img


def EdgePoints(image):
	# edge detection
	#em = feature.canny(img)
	em = sobel(image)
	cutoff = np.sqrt(4 * np.mean(em))
	em = feature.canny(em, high_threshold=cutoff)
	
	return np.asarray(np.nonzero(em))
	






# --------------- Ransac implementation (finish it!) ---------------


def Ransac(edge_points):
	# --- Parameters ---
	# TODO: minimum sample size for initial model fitting of a circle
	# s = ...
	max_iters = 100# maximum number of RANSAC iterations
	p = 0.90# probability for outlier-free sample after N iterations
	distance_threshold = 5# acceptable margin around model circle for inliers
	refit_threshold = 10# sample threshold for re-fitting (more inliers than this -> re-fit model params)
	radius_bounds = [4, 30]# lower and upper bound for radii
	center_bounds = [[0, img.shape[0]], [0, img.shape[1]]]# boundaries for center coordinates (i.e. image boundaries)
	num_points = edge_points.shape[1]# total number of edge points



	# --- return values ---
	num_inliers_best = 0 # number of inliers in currently best fitting model	
	center_best = None# x-, y-coordinate and radius of the best fitting model
	radius_best = None
	inliers_best = None # list of inlier samples corresponding to the best fitting model
	
	current_iter = 0
	N = max_iters
	
	while current_iter < max_iters and current_iter < N:
		print("iter %d, N = %d\n" % (current_iter, N))
		# draw s initial sample points
		# TODO: pick s edge points at random
		# initial_samples = ...
		
		# fit model to initial sample points
		[center, radius] = FitCircle(initial_samples)
		
		# discard models whose parameters are outside of the desired boundaries
		if radius < radius_bounds[0] or \
			radius_bounds[1] < radius or \
			center[0] < center_bounds[0][0] or \
			center_bounds[0][1] <= center[0] or \
			center[1] < center_bounds[1][0] or \
			center_bounds[1][1] <= center[1]:
				continue
		
		# count inliers	
		# TODO: find all edge points lying on the circle within the specified margin
		# distance_to_center = ...
		
		# TODO: compare with model radius to determine if points are inliers or not (within margin)
		# inliers = ...
		num_inliers = inliers.shape[1]
		
		# we optionally re-fit the model parameters if we have enough inliers
		if num_inliers > refit_threshold:
			[center, radius] = FitCircle(inliers)
		
		# did we find a better model than the previous one?
		if num_inliers_best < num_inliers:
			num_inliers_best = num_inliers
			center_best = center
			radius_best = radius
			inliers_best = inliers
	
		# update N according to the strategy in the slides
		# TODO: find fraction of outliers
		# eps = ...
		
		# TODO: compute N
		# N = ...
		
		
		
		
		current_iter += 1
		
	# New samples:
	# TODO: remove inliers without edge_points 
	# new_edgepoints = ...
	
	return (num_inliers_best, center_best, radius_best, inliers_best, new_edgepoints)
	

plt.close('all')
	
img = LoadImage("coins.png")
ep = EdgePoints(img)
ep_min = len(ep)/4


plt.figure("Edge Points")
plt.title("Edge Points")
plt.imshow(img, cmap=plt.cm.gray)
plt.scatter(ep[1, :], ep[0, :], s=1)


centers_x = []
centers_y = []
radii = []

for _ in range(10):
	
	(num_inliers_best, center_best, radius_best, inliers_best, new_edgepoints) = Ransac(ep)
	
	centers_x.append(center_best[1])
	centers_y.append(center_best[0])
	radii.append(radius_best)
	
	# display results
	plt.figure()
	plt.imshow(img, cmap=plt.cm.gray)
	plt.scatter(ep[1, :], ep[0, :], s=1)
	ax = plt.gca()
	DrawCircle(ax, [center_best[1]], [center_best[0]], [radius_best])
	ax.scatter(inliers_best[1, :], inliers_best[0, :], s=10, c='C1', linewidths=0)
	ax.scatter(center_best[1], center_best[0])
	
	ep = new_edgepoints

# display all circles
plt.figure("Final")
plt.imshow(img, cmap=plt.cm.gray)
plt.scatter(ep[1, :], ep[0, :], s=1)
ax = plt.gca()
DrawCircle(ax, centers_x, centers_y, radii)