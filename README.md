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
