# -*- coding: utf-8 -*-
"""
@author: Lennard Alms, Markus Laubenthal, Bilal Kizilkaya
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage import feature, io, draw
from skimage.filters import sobel
from math import pi, radians, cos, sin

# for comparisson
import scipy.ndimage as ndim
from skimage.transform import hough_circle
from scipy.stats import multivariate_normal

# Implement a function that, given an edge image and a range of radii, performs the Circular Hough Transform to create an accumulation array containing the votes for circles centered at the specific pixel and with a specific radius.
def hough_circles(edge_map, radius_range):
	# you can compare you solution, to a reference implementation:
	#acc = hough_circle(edge_map, radius_range) # <- remove this
	angles = np.arange(0, (2. - 2. / 360) * np.pi, 1. / 360)

	print(edge_map.shape)

	acc = np.zeros((radius_range.shape[0], edge_map.shape[0], edge_map.shape[1])).astype(np.float32)

	for i, radius in enumerate(radius_range):
		print(i)
		with np.nditer(edge_map, flags=['multi_index']) as it:
			for px in it:
				if px > 0:
					rr, cc= draw.circle_perimeter(it.multi_index[1],it.multi_index[0],radius)
					rr_FilterIndex = np.where((rr >= edge_map.shape[1]) | (rr < 0))
					cc_FilterIndex = np.where((cc >= edge_map.shape[0]) | (cc < 0))
					val_FilterIndex = np.append(rr_FilterIndex, cc_FilterIndex)
					rr = np.delete(rr, val_FilterIndex)
					cc = np.delete(cc, val_FilterIndex)

					#print(rr, cc)

					acc[i,cc,rr] += 1

					#skimage.draw
					#acc[i] += (radius * np.cos(angles) + it.multi_index[0], radius * np.sin(angles) + it.multi_index[1])
		print(acc[i])
	acc /= acc.max()


	# TODO: remove above line and implement your own Hough Transformation:
	# acc = ...
	return acc

# Implement this function for displaying four different slices for specific radii from the accumulation array. The first argument is the Hough accumulation array, the second argument is the same range of radii that was used to create the acculuation array.
def display_acc_slices(acc, radius_range):
	# create figure and axes inside of this figure
	(fh, ((ax1, ax2), (ax3, ax4), (ax5, ax6))) = plt.subplots(nrows=3, ncols=2)
	# display four slices as images using imshow
	# - read up on the vmin and vmax parameters to set the same color scaling for each slice
	# - set a title for each sub plot that shows the corresponding radius

	fh.suptitle("Radii slices")
	idx = np.linspace(0, acc.shape[0]-1, 6, dtype=np.int32)

	ax1.imshow(acc[idx[0],:,:], vmin=0, vmax=0.5)
	ax2.imshow(acc[idx[1],:,:], vmin=0, vmax=0.5)
	ax3.imshow(acc[idx[2],:,:], vmin=0, vmax=0.5)
	ax4.imshow(acc[idx[3],:,:], vmin=0, vmax=0.5)
	ax5.imshow(acc[idx[4],:,:], vmin=0, vmax=0.5)
	ax6.imshow(acc[idx[5],:,:], vmin=0, vmax=0.5)


# displays a set of circles, specified by three arrays containing x- & y-coordinates, as well as circle radius; optionally a score can be specified that is visulalized via the circle color (brighter color == higher score)
def draw_circle(axes_handle, centers_x, centers_y, radii, scores=None):
	#axes_handle.hold(True)
	n = 100 # number of samples for drawing circles
	if scores is None:
		colors = np.repeat(np.reshape([0, 0, 1], (1, 3)), 10, axis=0)
	else:
		(smin, smax) = (np.min(scores), np.max(scores))
		# create color map
		colors = cm.hot(np.arange(0, 256))
		colors = colors[63 : 192, 0:3]
		# use scores as indices into colormap
		scores = np.int64(np.ceil(127. * ((scores - smin) / (smax - smin))))
		colors = colors[scores, :]

	# iterate over all circles that are passed
	for x, y, radius, color in zip(centers_x, centers_y, radii, colors):
		angles = np.arange(0, (2. - 2. / n) * pi, 1. / n)
		(xs, ys) = (radius * np.cos(angles) + x, radius * np.sin(angles) + y)
		axes_handle.plot(xs, ys, color=tuple(color), linewidth=5)
	fh.canvas.draw()


# read image & compute edge map
img = io.imread("coins.png")

# convert to gray image if necessary
if len(img.shape) > 2:
	img = img[:, :, 0]

# edge detection
#em = feature.canny(img)
em = sobel(img)
cutoff = np.sqrt(4 * np.mean(em))
em = feature.canny(em, high_threshold=cutoff)

# display image & edge map
(fh, (ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=True)
ax1.imshow(img, cmap=plt.cm.gray)
ax2.imshow(em, cmap=plt.cm.gray)
ax2.set_title("Edge map")
fh.canvas.draw()

# also show detected edge points on original image
edge_points = np.nonzero(em)
ax1.scatter(x=edge_points[1], y=edge_points[0], c='b', s=1, linewidths=0)
ax1.set_title("Circles")
fh.canvas.draw()



# fit circles
# total number of circles to extract
num_circles = 10

# range of radii to look for in Hough space
hough_radii = np.arange(20, 70, 1)

# perform CHT (this needs to be implemented above)
acc = hough_circles(em, hough_radii)

# let's inspect the accumulation array by displaying some slices (needs to be implemented above)
display_acc_slices(acc, hough_radii)
#
# implement this: find local maxima in the hough accumulation array acc and extract the corresponding circle parameters (i.e. center x- & y-coordinate and radius) as well as each circle's voting score
# this should become a (num_circles x 2) array, storing the center coordinates in its colums
centers = []
# an array with num_circles elements specifying the radius of each circle
radii = []
# an array with num_circles elements storing the voting score for each circle
scores = []

# perform slight smoothing in hough space
#acc = ndim.filters.gaussian_filter(acc, sigma=0.5)
#display_acc_slices(acc, hough_radii)

# find local maxima
# we subtract a 3D-gaussian centered around each local maximum to avoid extracting similar circles from nearby bins (not mandatory for solution)
[rs, xs, ys] = np.meshgrid(hough_radii, np.arange(0, em.shape[0]), np.arange(0, em.shape[1]), indexing='ij')
grid = np.empty(xs.shape + (3,))
grid[:, :, :, 0] = rs
grid[:, :, :, 1] = xs
grid[:, :, :, 2] = ys
for ii in np.arange(0, num_circles):
	# find current maximum
	(rad, y, x) = np.unravel_index(np.argmax(acc), acc.shape)
	# extract circle parameters corresponding to this maximum
	scores.extend([acc[rad, y, x]])
	centers.extend(np.array([x, y]).reshape(1, 2))
	radii.extend([hough_radii[rad]])
	# not mandatory for solution
	# we now want to get rid of this maximum to be able to find the next lower one
	# set up and evaluate 3D gaussian
	# uniform gaussian with standard deviation 0.1
	covariance = np.eye(3) * 10
	mvn = multivariate_normal([hough_radii[rad], y, x], covariance)
	acc_sub = mvn.pdf(grid)
	# scale gaussian
	acc_sub *= scores[-1] / np.max(acc_sub)
	# subtract gaussian from accumulation array
	acc -= acc_sub
	# a much simpler and faster acceptable solution is setting the local maxima to 0
	# this produces poor results, though
	#acc[rad, y, x] = 0


centers = np.array(centers)
radii = np.array(radii)
scores = np.array(scores)

# sort again by score to be sure
idxs = np.argsort(scores)
centers = centers[idxs[::-1], :]
radii = radii[idxs[::-1]]
scores = scores[idxs[::-1]]


#display results
draw_circle(ax1, centers[:, 0], centers[:, 1], radii, scores)
plt.show()
