#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:26:42 2020

@author: james
"""

import numpy as np
import mayavi.mlab as mlab


# Points
x, y, z, value = np.random.random((4, 40))
mlab.points3d(x, y, z, value)


# Lines
mlab.clf()  # clear the figure
t = np.linspace(0, 20, 200)
mlab.plot3d(np.sin(t), np.cos(t), 0.1*t, t)


# Elevation surface
mlab.clf()
x, y = np.mgrid[-10:10:100j, -10:10:100j]
r = np.sqrt(x**2 + y**2)
z = np.sin(r)/r
mlab.surf(z, warp_scale='auto')


# Arbitrary regular mesh
mlab.clf()
phi, theta = np.mgrid[0:np.pi:11j, 0:2*np.pi:11j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
mlab.mesh(x,y,z)
mlab.mesh(x,y,z, representation='wireframe', color=(0,0,0))


# Volumetric data
mlab.clf()
x,y,z = np.mgrid[-5:5:64j, -5:5:64j, -5:5:64j]
values = x*x*0.5 + y*y + z*z*2.0
mlab.contour3d(values)



# image show
mlab.imshow(np.random.random((10,10)))


# plot3d
mlab.test_plot3d()


# screenshot
mlab.test_plot3d()
arr = mlab.screenshot()
import pylab as pl
pl.imshow(arr)
pl.axis('off')
pl.show()









