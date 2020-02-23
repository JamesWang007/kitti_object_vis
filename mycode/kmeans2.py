# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:31:02 2020

@author: bejin
"""

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs



def initialize_clusters(points, k):
    """Initializes clusters as k randomly selected points from points."""
    return points[np.random.randint(points.shape[0], size=k)]
    
# Function for calculating the distance between centroids
def get_distances(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    return np.linalg.norm(points - centroid, axis=1)



def load_res_data(filename):
    data_list = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            #data = np.array([float(s) for s in line.split(" ")])
            data_list.append( np.fromstring(line, dtype=float, sep=' ') )
    return np.stack(data_list, axis = 0)


def get_car_pts():
    pts_res = load_res_data('res_03.txt')
    choice = np.where(pts_res[:,3]>0)
    pts_car = pts_res[choice, :].squeeze()
    pts_car_xy = pts_car[:, 0:2] 
    return pts_car_xy


# Generate dataset
X, y = make_blobs(centers=3, n_samples=500, random_state=1)
pts_car_xy = get_car_pts()

X = pts_car_xy

# Visualize
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(X[:,0], X[:,1], alpha=0.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$');



k = 6
maxiter = 50

# Initialize our centroids by picking random data points
centroids = initialize_clusters(X, k)

# Initialize the vectors in which we will store the
# assigned classes of each data point and the
# calculated distances from each centroid
classes = np.zeros(X.shape[0], dtype=np.float64)
distances = np.zeros([X.shape[0], k], dtype=np.float64)

# Loop for the maximum number of iterations
for i in range(maxiter):
    
    # Assign all points to the nearest centroid
    for i, c in enumerate(centroids):
        distances[:, i] = get_distances(c, X)
        
    # Determine class membership of each point
    # by picking the closest centroid
    classes = np.argmin(distances, axis=1)
    
    # Update centroid location using the newly
    # assigned data point classes
    for c in range(k):
        centroids[c] = np.mean(X[classes == c], 0)



group_colors = ['skyblue', 'coral', 'lightgreen', 'red', 'green', 'blue']
colors = [group_colors[j] for j in classes]

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(X[:,0], X[:,1], color=colors, alpha=0.5)
ax.scatter(centroids[:,0], centroids[:,1], color=['black'], marker='*', lw=2)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$');













