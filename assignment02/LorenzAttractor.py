import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint


def LorenzAttractor(t, X):
	"""
	Computes the force at a time t and a position X
	t: the current time. In this case, the function is constant in t
	X: the position as [x, y, z] array
	"""
	force = np.zeros((3))
	sigma = 10
	p = 28
	beta = 8/3

	force[0] = sigma * (X[1] - X[0])
	force[1] = X[0] * (p - X[2]) - X[1]
	force[2] = X[0] * X[1] - beta * X[2]

	return force

def Euler(f, y0, stepsize, steps):
	"""
	Compute positions using the Euler integration scheme.
	f:			The function that is being integrated
	y0:			The start position
	stepsize:	Delta for each step
	steps:		Amount of integration steps
	"""
	path = np.zeros((3, steps))
	pos = y0
	for t in range(steps):
		force = f(t, pos)
		pos += stepsize * force
		path[0][t] = pos[0]
		path[1][t] = pos[1]
		path[2][t] = pos[2]

	return path

def RungeKutta4thOrder(f, y0, stepsize, steps):
	"""
	Compute positions using the Runge-Kutta 4th Order integration scheme.
	f:			The function that is being integrated
	y0:			The start position
	stepsize:	Delta for each step
	steps:		Amount of integration steps
	"""
	x = np.zeros((3, steps))
	# your code goes here...
	return x


# Compute Values
y0 = [-1, 3, 4]
euler = Euler(LorenzAttractor, y0, 0.025, 3000)
rungekutta = RungeKutta4thOrder(LorenzAttractor, y0, 0.025, 3000)
ref = odeint(lambda y, t : LorenzAttractor(t, y), y0, np.arange(0, 0.025*3000, 0.025)).T


# Plot Everything
plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
ax.set_title("Euler Integration")
ax.plot(euler[0, :], euler[1, :], euler[2, :])

plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
ax.set_title("Runge Kutta Integration")
ax.plot(rungekutta[0, :], rungekutta[1, :], rungekutta[2, :])

plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
ax.set_title("ODE Integrate")
ax.plot(ref[0, :], ref[1, :], ref[2, :])

plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
ax.set_title("Comparison")
ax.plot(euler[0, :], euler[1, :], euler[2, :], label="Euler")
ax.plot(rungekutta[0, :], rungekutta[1, :], rungekutta[2, :], label="Runge Kutta")
ax.plot(ref[0, :], ref[1, :], ref[2, :], label="ODE Integrate")
plt.legend()

plt.show()
