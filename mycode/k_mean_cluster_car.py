# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:28:20 2020

@author: bejin
"""
import numpy as np
import _init_path
import mayavi.mlab as mlab


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

def kmeans(data, k=3, normalize=False, limit=500):
    """Basic k-means clustering algorithm.
    link: https://ixora.io/itp/learning_machines/clustering-and-numpy/
    """
    # optionally normalize the data. k-means will perform poorly or strangely if the dimensions
    # don't have the same ranges.
    if normalize:
        stats = (data.mean(axis=0), data.std(axis=0))
        data = (data - stats[0]) / stats[1]
    
    # pick the first k points to be the centers. this also ensures that each group has at least
    # one point.
    centers = data[:k]

    for i in range(limit):
        # core of clustering algorithm...
        # first, use broadcasting to calculate the distance from each point to each center, then
        # classify based on the minimum distance.
        classifications = np.argmin(((data[:, :, None] - centers.T[None, :, :])**2).sum(axis=1), axis=1)
        # next, calculate the new centers for each cluster.
        new_centers = np.array([data[classifications == j, :].mean(axis=0) for j in range(k)])

        # if the centers aren't moving anymore it is time to stop.
        if (new_centers == centers).all():
            break
        else:
            centers = new_centers
    else:
        # this will not execute if the for loop exits on a break.
        raise RuntimeError(f"Clustering algorithm did not complete within {limit} iterations")
            
    # if data was normalized, the cluster group centers are no longer scaled the same way the original
    # data is scaled.
    if normalize:
        centers = centers * stats[1] + stats[0]

    print(f"Clustering completed after {i} iterations")

    return classifications, centers



# Function for calculating the distance between centroids
def get_distances(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    return np.linalg.norm(points - centroid, axis=1)



class Cluster():
    def __init__(self, id, x=0, y=0, z=-1, cls=None, pts=None, n=0):
        self.id = id
        self.x = x
        self.y = y 
        self.z = z
        self.cls = cls
        self.pts = pts
        self.n = n        
        #self.get_z()
        
        
    def get_z(self):
        self.z = np.mean(self.pts[:,2])
        return self.z
        
    def get_max_min(self):
        x_max = np.max(self.pts[:, 0])
        x_min = np.min(self.pts[:, 0])
    
        y_max = np.max(self.pts[:, 1])
        y_min = np.min(self.pts[:, 1])
        
        self.get_z()
        z_max = np.max(self.pts[:, 2])
        z_max = np.min(self.pts[:, 2])
        
        # h, w, l
        return np.min(x_max-x_min, 4), np.min(y_max-y_min, 4), np.min(z_max-z_min, 4)
            
    '''
    def get_center_vector():
        # 8-dim: <id, x, y, z, w, h, l, n>
        return np.array([self.id, self.x, self.y, self.z, 
                         ])
     '''  
        
class Group():
    # a group of clusters
    
    def __init__(self, pts, classifications, centers):
        # pts: (N, 3)
        # classifications: (N,); labels of points
        # centers: (K,2); K centers 
        
        self.cls = classifications
        self.centers = centers
        self.clusters = []
        self.pts = pts
        self.min_distance = 2.0
        
    # better run it after: self.merge_clusters
    def generate_clusters(self):  
        for i, C in enumerate(self.centers):
            # 8 dim : <id, x, y, z, w, h, l, n>
            id = i
            x = C[0]
            y = C[1]    
            one_cluster_cls = np.where(self.cls == i)
            one_cluster_pts = self.pts[one_cluster_cls]
            n = one_cluster_cls[0].shape[0]
            
            cluster = Cluster(id, x, y, -1.0, cls = one_cluster_cls, pts = one_cluster_pts, n=n)
            self.clusters.append(cluster)
        return self.clusters 
        
    
    def merge_clusters(self):
        if self.centers is None:
            return None
        merged_centers = []
        flag_clusters = np.ones((self.centers.shape[0]))
        for i, C in enumerate(self.centers):
            if flag_clusters[i]:
                flag_clusters[i] = 0
                merged_centers.append(C)
                distances = get_distances(self.centers, C)
                for j, d in enumerate(distances):
                    if d < self.min_distance and flag_clusters[j]:
                        self.cls[np.where(self.cls == j)] = i
                        #self.centers[j] = C
                        flag_clusters[j] = 0
            else:
                continue
        self.centers = merged_centers
        return self.cls, self.centers
    
    
def clustering_kmeans(pts, K=5):
    # clustering
    
    cls, C = kmeans(pts, K)
    
    return cls, C


def operation_clu(pts_car, pts_car_xy):
    K = 5
    classifications, centers = clustering_kmeans(pts_car_xy)
    group = Group(pts_car, classifications, centers)
    _, centers = group.merge_clusters()
    centers_4d_2 = np.concatenate( (centers, np.ones((len(centers), 2))*(-1)), axis=1 )
    
    return group

    
def main():
     # timer start
    from timeit import default_timer as timer
    start = timer()
    
    ## load my data
    #pts_car_xy = get_car_pts()
    pts_res = load_res_data('res_03.txt')
    choice = np.where(pts_res[:,3]>0)
    pts_car = pts_res[choice, :].squeeze()
    pts_car_xy = pts_car[:, 0:2]   
    
    
    pts_part_01 = pts_car_xy[np.where(pts_car_xy[:, 0] < 20)]
    pts_part_02 = pts_car_xy[np.where(pts_car_xy[:, 0] > 20)]
    
    
    g1 = operation_clu(pts_car, pts_part_01)
    g1.generate_clusters()
    features_xyzn_1 = [[c.x, c.y, c.get_z(), np.int(c.n)] for c in g1.clusters]
    
    
    g2 = operation_clu(pts_car, pts_part_02)
    g2.generate_clusters()
    features_xyzn_2 = [[c.x, c.y, c.get_z(), np.int(c.n)] for c in g2.clusters]

    features_xyzn = np.concatenate((features_xyzn_1, features_xyzn_2), axis = 0)

    ids = np.array(range(len(features_xyzn)), dtype=np.int).reshape(-1,1)
    whl = np.ones((len(features_xyzn), 3))
    whl[:, 0:1] = whl[:, 0:1] * 4
    whl[:, 2] = whl[:, 2] * 4
    
    res = np.concatenate((ids, features_xyzn, whl), axis=1)
    
    # kmeans 
    #K = 5   # predefined number of clusters
    #classifications, centers = kmeans(pts_car_xy, K)
    #classifications, centers = clustering_kmeans(pts_part_01)
    #pts_car_cls = np.concatenate((pts_car, classifications.reshape(-1,1)), axis=1)
    #centers_4d = np.concatenate( (centers, np.ones((K, 2))*(-1)), axis=1 )
    
    
    #create group of clusters
    #group = Group(pts_car_xy, classifications, centers)
    #_, centers = group.merge_clusters()
    #centers_4d_2 = np.concatenate( (centers, np.ones((len(centers), 2))*(-1)), axis=1 )
    
    fig = draw_lidar(pts_car)
    mlab.points3d(features_xyzn[:, 0], features_xyzn[:, 1], features_xyzn[:, 2], color=(0.5, 0.5, 1), mode="sphere", scale_factor=0.8)
    
    end = timer()
    print("cost time: %6f", end - start)

if __name__ == "__main__":
    main()