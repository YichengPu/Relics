from collections import defaultdict
import numpy as np

def kmeans(n_iter,K,train_x):
    start_centroids_index=np.random.choice(np.arange(len(train_x)),K,replace=False)

    first_centroid=np.random.choice(np.arange(len(train_x)),1,replace=False)
    start_centroids=[train_x[first_centroid,:]]

    while len(start_centroids)<K:
        D_weights=[]
        for i in range(len(train_x)):
            min_dist=np.inf
            for j in range(len(start_centroids)):
                curr_dist=np.linalg.norm(train_x[i,:]-start_centroids[j])
                min_dist=min(curr_dist,min_dist)
            D_weights.append(min_dist**2)
        D_weights=np.array(D_weights)/sum(D_weights)

        next_centroid=np.random.choice(np.arange(len(train_x)),1,replace=False,p=D_weights)
        start_centroids.append(train_x[next_centroid,:])

    start_centroids=np.array(start_centroids)
    # start_centroids=train_x[start_centroids_index]
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
