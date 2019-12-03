import matplotlib.pyplot as plt
import matplotlib.image
import scipy.signal
import numpy as np


# this is a convenience function for nicer image plotting with adjusted colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plotImg(img, title=None, ax=None, **more):
	if ax is None:
		ax=plt.gca()
	i = ax.imshow(img, **more)
	ax.set_xticks([])
	ax.set_yticks([])
	if title is not None:
		ax.set_title(title)
	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(i, cax=cax)
	return ax

# Part A
def MyConvolve2d(f, g):
	c = np.array([[0]])
	# TODO: Implement image convolution and return the result
	# WARNING: Make sure that your solution is efficient, see sheet for details

	return c


# Part B
def MyFftConcolve2d(f, g):
	c = np.array([[0]])

	# TODO: Implement image convolution using fft and return the result
	# Hints:
	# - use np.fft.fft2 and np.fft.ifft2
	# - The Fourier transform generates complex-valued matrices that may require further operations for displaying.

	return c

def PartA():
	k = matplotlib.image.imread('kernel.png')[:, :, 0]
	i = matplotlib.image.imread('img.png')[:, :, 0]
	c = scipy.signal.convolve2d(i, k, boundary='wrap')
	m = MyConvolve2d(i, k)


	plt.figure("Convolution by Hand")

	plt.subplot(141)
	plotImg(i, "Original Image", interpolation="nearest", cmap=plt.cm.gray)

	plt.subplot(142)
	plotImg(c, "Python Convolution", interpolation="nearest", cmap=plt.cm.gray)

	plt.subplot(143)
	plotImg(m, "Convolution by Hand", interpolation="nearest", cmap=plt.cm.gray)

	if m.shape != c.shape:
		print("Image Dimensions do not match!")
	else:
		plt.subplot(144)
		plotImg(m-c, "Difference", interpolation="nearest", cmap=plt.cm.gray)

	plt.show()



def PartB():
	i = matplotlib.image.imread('img.png')[:, :, 0]
	k = matplotlib.image.imread('kernel.png')[:, :, 0]
	c = scipy.signal.convolve2d(i, k, boundary='wrap')
	m = MyFftConcolve2d(i, k)


	plt.figure("Convolution with FFT")

	plt.subplot(141)
	plotImg(i, "Original Image", interpolation="nearest", cmap=plt.cm.gray)

	plt.subplot(142)
	plotImg(c, "Python Convolution", interpolation="nearest", cmap=plt.cm.gray, vmax=32)

	plt.subplot(143)
	plotImg(m, "FFT Convolution", interpolation="nearest", cmap=plt.cm.gray, vmax=32)

	if m.shape != c.shape:
		print("Image Dimensions do not match!")
	else:
		plt.subplot(144)
		plotImg(m-c, "Difference", interpolation="nearest", cmap=plt.cm.gray)

	plt.show()

PartA()
PartB()
