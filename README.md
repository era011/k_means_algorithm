A function implementing the K-Means algorithm has been written.

## k-means algorithm

The k-means algorithm is one of the most popular data clustering methods. It is used to partition a dataset into (k) groups (clusters) based on data similarity. This algorithm is especially useful in tasks involving segmentation, data simplification, and pattern analysis.
Basic steps of the k-means algorithm:
1) ### Selecting the number of clusters:
      * At the first stage, you need to set the number of clusters 
, into which the data set will be divided. This parameter is usually selected experimentally.
2) ### Initializing centroids:

      * Randomly selected 
 points from the data that will become the initial centroids of the clusters. The centroid is the point representing the center of the cluster.
3) ### Assigning points to clusters:

      * For each point in the data set, the distance to each of the 
 centroids. A point is assigned to the cluster whose centroid is closest to that point. This step typically uses Euclidean distance, but other distance metrics can be used.
4) ### Centroid update:

      * After assigning all points, the centroids are recalculated. The new centroid of each cluster is calculated as the average of all points belonging to that cluster.
5) ### Repeat steps 3 and 4:

      * Steps 3 and 4 are repeated until the centroids stop changing (or the changes are insignificant, less than $\epsilon$ ) or until the maximum number of iterations is reached. This means that the algorithm has converged and the clusters have stabilized.
## Conclusion
The k-means algorithm is a powerful tool for data clustering, which is used in various fields such as marketing, bioinformatics, image processing and many others. Despite its limitations, it remains one of the main methods for solving clustering problems.

## Code

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def distance(points:np.ndarray,centroids:np.ndarray,dot:int)->np.ndarray:
    vector=centroids[dot]
    vectors=points
    return np.linalg.norm(vectors-vector,axis=1)




def kmeans(points:np.ndarray,count_classes:int,epsilon:float=.05,count_of_iterations:int=0)->np.ndarray:

    dimension=points.shape

    # берем индексы точек сулайным образом
    index_of_centroinds=np.random.randint(0,points.shape[0],size=count_classes)

    distances=distance(points,points[index_of_centroinds],0)
    centroids=points[index_of_centroinds]
    distances=distances[:,np.newaxis]



    if count_of_iterations!=0:
        for j in range(count_of_iterations):

            # строим матрицу расстояний до классов
            if distances.shape[1]<count_classes:
                for i in range(1,count_classes):
                    distances=np.hstack((distances,distance(points,centroids,i)[:,np.newaxis]))  
            else:
                for i in range(count_classes):
                    distances[:,i]=distance(points,centroids,i)

            # находим индексы ближайших точек до центроидов
            index_min_distances=np.argmin(distances,axis=1)
            index_min_distances=index_min_distances[:,np.newaxis]    

            # обновляем метки точек
            if points.shape[1]>dimension[1]:
                points=np.delete(points, -1, axis=1)
            points=np.concatenate((points,index_min_distances),axis=1)

            # обновляем центроиды
            for i in range(count_classes):
                if i in np.unique(points[:,-1]):
                    centroids[i]=points[points[:,-1]==i][:,:-1].mean(axis=0)
    else:
        index_of_old_centroinds=np.random.randint(0,points.shape[0],size=count_classes)
        old_centroids=points[index_of_old_centroinds]
        while not np.all(np.linalg.norm((old_centroids-centroids),axis=1)<epsilon):

            # строим матрицу расстояний до классов
            if distances.shape[1]<count_classes:
                for i in range(1,count_classes):
                    distances=np.hstack((distances,distance(points,centroids,i)[:,np.newaxis]))  
            else:
                for i in range(count_classes):
                    distances[:,i]=distance(points[:,:-1],centroids,i)

            # находим индексы ближайших точек до центроидов
            index_min_distances=np.argmin(distances,axis=1)
            index_min_distances=index_min_distances[:,np.newaxis]    

            # обновляем метки точек
            if points.shape[1]>dimension[1]:
                points=np.delete(points, -1, axis=1)
            points=np.concatenate((points,index_min_distances),axis=1)

            # обновляем центроиды
            old_centroids=centroids.copy()
            for i in range(count_classes):
                if i in np.unique(points[:,-1]):
                    centroids[i]=points[points[:,-1]==i][:,:-1].mean(axis=0)

    return points
```
The kmeans function gets
* points - matrix of dimension NxM, where N is the number of points, M is the dimension of the space to which it belongs
* count_of_classes - the number of classes to divide into
* epsilon - for stop conditions
* count_of_iterations - number of iterations
returns an NxM+1 matrix, where the last column is the class labels

## How to use example:

```python
coords = np.concatenate([np.random.randn(100, 3)+np.array([4, 4,4]),
                         np.random.randn(100, 3)+np.array([4, -4, 4]),
                         np.random.randn(100, 3)+np.array([-4, 4,4]),
                         np.random.randn(100, 3)+np.array([-4, -4,4]),
                         np.random.randn(100, 3)+np.array([0, 0,0])], axis=0)

k=5


coords=kmeans(coords,count_classes=k)

fig1=go.Figure(data=go.Scatter3d(x=coords[:,0],y=coords[:,1],z=coords[:,2],mode='markers'))

clas=[]
for i in np.unique(coords[:,-1]):
    dp=coords[coords[:,-1]==i]
    clas.append(go.Scatter3d(x=dp[:,0],y=dp[:,1],z=dp[:,2],mode='markers'))

fig=go.Figure(data=clas)  
fig.write_html('after.html')  
fig1.write_html('before.html')
```
