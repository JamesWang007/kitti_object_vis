# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:50:37 2020

@author: bejin
"""


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



from sklearn.cluster import DBSCAN
import numpy as np
X = get_car_pts()

clustering = DBSCAN(eps=3, min_samples=2).fit(X)
clustering.labels_

clustering