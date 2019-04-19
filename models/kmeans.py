from collections import defaultdict
import numpy as np

def kmeans(n_iter,K,train_x):
    start_centroids_index=np.random.choice(np.arange(len(train_x)),K)
    start_centroids=train_x[start_centroids_index]
    centroids=start_centroids

    for j in range(n_iter):
        cluster_assignment=defaultdict(set)
        for i in range(len(train_x)):
            min_dist=np.inf
            n=0
            for center in centroids:
                dist=np.linalg.norm(center-train_x[i,:])
                if dist<min_dist:
                    min_dist=dist
                    min_center_index=n
                n+=1
            cluster_assignment[min_center_index].add(i)

        for cluster in cluster_assignment.keys():
            points=cluster_assignment[cluster]
            new_centroid=np.mean(train_x[list(points)],axis=0)
            centroids[cluster]=new_centroid

    return cluster_assignment
