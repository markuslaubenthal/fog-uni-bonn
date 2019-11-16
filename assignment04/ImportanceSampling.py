# Authors: Markus Laubenthal, Bilal Kizilkaya, Lennard Alms


import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, pi


# f: target function, which we want to integrate
def f(x):
    return np.power(x, 0.9) * np.exp( -(x ** 2) / 2)



# 2a) FoG Add a short comment on how you would expect the three different distributions to perform and why:
# We expect the p_p to perform the best because it has the smallest variance w.r.t f(x)

# p_n: normal distribution for samples
sigma2 = 0.3
mu = 1.2
def p_n(x):
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-(mu - x)**2/(2 * sigma2))

# p_p: polynomial fit of the target function
# ...
# (make sure, the fitting is computed only once and not on every invocation of p_p !
_p_p = None
_degree = 4
_interval = np.arange(0, 4, 0.1)
def p_p(x):
    global degree
    global _p_p
    global _interval
    if(_p_p is None):
        y = f(_interval)
        _p_p = np.polyfit(_interval, y, _degree)
    return np.polyval(_p_p, x)

# plot the function graphs:
plt.figure("Functions", figsize=(9,3))
plt.subplot(131)
plt.ylim(0,0.8)
plt.title(r'f(x)')
plt.plot(_interval, f(_interval))
plt.subplot(132)
plt.ylim(0,0.8)
plt.title(r'p_n(x)')
plt.plot(_interval, p_n(_interval))
plt.subplot(133)
plt.ylim(0,0.8)
plt.title(r'p_p(x)')
plt.plot(_interval, p_p(_interval))
plt.show()




"""
Uses rejection sampling to generate samples
n: amount of samples to generate
d: distribution to draw samples from
max: maximum y value that is used to generate the samples
returns: x and y values of the samples in the shape (n, 2)
"""
def GenSamples(n, d, max):
    s = np.zeros((n, 2))
    counter = 0
    while(counter < n):
        sample = np.random.rand(2)
        sample[0] *= 4
        sample[1] *= 0.8
        val = d(sample[0])
        if(sample[1] < val):
            s[counter] = sample
        counter += 1
    return s


# Plot results of GenSamples()
# Hint: 0.8 is a reasonable value for the max parameter of GenSamples

function_x = np.linspace(0, 4, 100)

plt.figure("Normal Distribution")
samples = GenSamples(200, p_n, 0.8)
plt.scatter(samples[:,0], samples[:,1])
plt.plot(function_x, p_n(function_x))
plt.show()

plt.figure("Polynomial")
samples = GenSamples(200, p_p, 0.8)
plt.scatter(samples[:,0], samples[:,1])
plt.plot(function_x, p_p(function_x))
plt.show()

"""
p: the function to integrate
samples: array with the sample positions
weights: function to compute the weight of each sample
"""
def Integrate(p, samples, weights):
    'integrier dich!'
    sample_result = p(samples) * (4) * weights
    I = np.sum(sample_result) * (1/len(samples))
    return I



maximumSamples = 500
id = np.zeros(maximumSamples)
norm = np.zeros(maximumSamples)
poly = np.zeros(maximumSamples)

for i in range(1,maximumSamples):
    weights = None
    samples = GenSamples(i, f, 0.8)
    id[i] = Integrate(f, samples[:,0], f(samples[:,0]))
    samples = GenSamples(i, p_n, 0.8)
    norm[i] = Integrate(f, samples[:,0], p_n(samples[:,0]))
    samples = GenSamples(i, p_p, 0.8)
    poly[i] = Integrate(f, samples[:,0], p_p(samples[:,0]))
    if i%10 == 0: # print progress
        print(i)



plt.figure("Convergence", figsize=(12,12))
plt.subplot(311)
plt.ylim(0,0.6)
plt.title(r'f(x)')
plt.plot(range(maximumSamples), id)
plt.subplot(312)
plt.ylim(0,0.6)
plt.title(r'p_n(x)')
plt.plot(range(maximumSamples), norm)
plt.subplot(313)
plt.ylim(0,0.6)
plt.title(r'p_p(x)')
plt.plot(range(maximumSamples), poly)
plt.show()
