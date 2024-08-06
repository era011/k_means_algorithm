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




# coords = np.concatenate([np.random.randn(100, 3)+np.array([4, 4,4]),
#                          np.random.randn(100, 3)+np.array([4, -4, 4]),
#                          np.random.randn(100, 3)+np.array([-4, 4,4]),
#                          np.random.randn(100, 3)+np.array([-4, -4,4]),
#                          np.random.randn(100, 3)+np.array([0, 0,0])], axis=0)

# coords=kmeans(coords,count_classes=5,epsilon=0.005)

# coords=kmeans(coords,count_classes=k)

# print(coords)
# for i in range(k):
#     print(f'количество точек в {i} классе {coords[coords[:,-1]==i].shape[0]}')