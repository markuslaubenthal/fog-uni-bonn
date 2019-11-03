#Basic ray tracing rendering pipeline including reflection and shading model.
import numpy as np
import matplotlib.pyplot as plt
import sys



#width, height = (1000, 700)
width, height = (400, 280)
eps = .0001


def report_progress(current, total):
    sys.stdout.write('\rProgress: {:.2%}'.format(float(current)/total))
    if current==total:
       sys.stdout.write('\n')
    sys.stdout.flush()

def normalizer(x):
    x /= np.linalg.norm(x)
    return x

def intersectPlane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the plane (P, N).
    norm = np.dot(D, N)
    if np.abs(norm) < 1e-10:
        return 1e100
    d = np.dot(P - O, N) / norm
    if d < 0:
        return 1e100
    return d

def intersectSphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the sphere (S, R).
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return 1e100

def intersection(O, D, obj):
    """
    Returns:
        d: distance to intersection point, 1e100 if nothing was hit
    """
    if obj['type'] == 'plane':
        return intersectPlane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersectSphere(O, D, obj['position'], obj['radius'])

def getNormal(obj, M):
    if obj['type'] == 'sphere':
        N = normalizer(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    return N

def getColor(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

def rayTracing(rayOrigin, rayDir):
    """Finding the first hitting point of the ray with objects in the scene.

    Returns (if nothing was hit):
        'None'
    Returns (if something was hit):
        obj: the object that was hit (as a dictionary)
        M: the intersection point
        N: the normal at the intersection point
        color_ray: diffuse shaded color at the intersection point
    """

    hit = 1e100
    for i, obj in enumerate(scene):
        hitDistance = intersection(rayOrigin, rayDir, obj)
        if hitDistance < hit:
            hit, thatObj = hitDistance, i
    # Return None if the ray does not hit any object.
    if hit == 1e100:
        return None

    # Finding which object it was.
    obj = scene[thatObj]

    # Calculating the point of intersection on the object.
    M = rayOrigin + rayDir * hit
    N = getNormal(obj, M)
    color = getColor(obj, M)
    DirtoLight = normalizer(L - M)
    DirtoCamera = normalizer(O - M)

    ##########################################################
    ##TODO
    # part a
    # Check whether the point is shadowed or visible.
    lightFactor = 1 # lightFactor=0 means this object is in the shadow

    hit = 1e100
    for i, obj in enumerate(scene):
        hitDistance = intersection(M + N * 0.0001, DirtoLight, obj)
        if hitDistance < hit:
            hit, thatObj = hitDistance, i
    if hit < 1e100:
        lightFactor = 0

    ###########################################################
    # Computing the color, the default is ambient.
    color_ray = 0
    color_ray += color * lightFactor * obj.get('ambient', ambient)

    ##################################################################################################################
    ##TODO
    # part c
    # Lambert shading (diffuse part).
    # Blinn-Phong shading (specular part).
    ##################################################################################################################
    color_ray += obj.get('diffuse_c', diffuse_c) * np.dot(N, DirtoLight) * color * lightFactor

    color_ray += obj.get('specular_c', specular_c) * np.dot(N, normalizer(DirtoLight + DirtoCamera)) ** obj.get('specular_k', specular_k) * color_light


    return obj, M, N, color_ray

def addSphere(position, radius, color):
    return dict(type='sphere', position=np.array(position),
        radius=np.array(radius), color=np.array(color),
        ambient=.05, diffuse_c=.5, specular_c=.5, specular_k=50, reflection=.5)

def addPlane(position, normal, color):
    return dict(type='plane', position=np.array(position),
        normal=np.array(normal), color=np.array(color),
        ambient=.05, diffuse_c=.5, specular_c=.5, specular_k=50, reflection=.5)

# Objects in the scene (three spheres and one infinite plane).
scene = [addSphere([.7, .1, 1.], .7, [0., 0., 1.]),
         addSphere([-.7, .1, 2.2], .7, [0., 1., 0.]),
         addSphere([-2.7, .1, 3.5], .7, [1., 0., 0.]),
         addPlane([0., -.75, 0.], [0., 1., 0.], [1., 1., 1.]),
         ]

# Light position and its color.
L = np.array([5., 5., -10.])
color_light = np.ones(3) # white

# Material parameters.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

hit_max = 4  # Maximum number of light reflection.
color = np.zeros(3)  # Initialize current color.
O = np.array([0., 0.2, -1.])  # Camera position (origin of rays).
Q = np.array([0., 0., 0.])  # Camera pointing toward.
img = np.zeros((height, width, 3)) # Initialize RGB image

# Screen: x0, y0, x1, y1.
W = width / (height*1.0)
Sc = (-1., -1. / W + .2, 1., 1. / W + .2)

# Loop through all pixels.
for i, x in enumerate(np.linspace(Sc[0], Sc[2], width)):
    for j, y in enumerate(np.linspace(Sc[1], Sc[3], height)):
        color[:] = 0
        Q[:2] = (x, y) # point on camera plane
        D = normalizer(Q - O)
        hit = 0
        rayOrigin, rayDir = O, D
        reflection = 1.
        # Loop through initial and reflected rays.
        while hit < hit_max:
            rayTraced = rayTracing(rayOrigin, rayDir)
            if rayTraced == None:
                break;
            obj, M, N, color_ray = rayTraced
            #####################################################################
            ##TODO
            # part b
            # Reflection can be implemented by create a new ray.
            #####################################################################
            rayOrigin = M + N * 0.0001
            rayDir = normalizer(rayDir - 2 * np.dot(rayDir, N) * N)

            hit += 1
            color += reflection * color_ray
            reflection *= obj.get('reflection')
        img[height - j - 1, i, :] = np.clip(color, 0, 1)
    report_progress((i+1)*height, width*height)

plt.imsave('scene.png', img)
