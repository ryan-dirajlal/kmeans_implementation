
import random
import numpy as np


# Sources: 
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# https://pythonguides.com/python-numpy-shape/

class KMeans_implementation():
    
    def __init__(self, n_clusters=4, max_iter=1000, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter=max_iter
        self.random_state=random_state
        self.cluster_centers_ = None # numpy array # see create_random_centroids for more info
        self.labels_ = None # predictions # numpy array of size len(input)

    def fit(self, input: np.ndarray) -> np.array: 
        # Fitting a model means to train the model on some data using my specific algorithm. 
        # This function will simply return the cluster centers, but it will also update the cluster centers and predictions.
        num_features = np.shape(input)[1] # num of dimensions and values
        self.labels_ = np.array([0] * np.shape(input)[0])
        self.cluster_centers_ = self.init_centroids(num_features, input)

        for i in range(self.max_iter):
            old_cluster_centers = self.cluster_centers_.copy() #start off random (see init_centroids)
            #Assigns df points to center of cluster
            self.group_clusters(input)
            self.recenter_centroids(input)

        return self.cluster_centers_


    def init_centroids(self, num_features: int, input: np.ndarray) -> np.array:
        #creates random starting points for centroids
         #cluster_centers_ is an attribute that I will update.
        self.cluster_centers_= np.random.rand(self.n_clusters,num_features) #will have n_clusters amount 
        
    
        
        return None

    def calculate_distance(self, d_features, c_features) -> int:
        #Calculates the Euclidean distance between point A and point B. 
        
        distance = np.sum(np.square(d_features - c_features))

        return distance

    def recenter_centroids(self, input: np.array) -> None:
        # This function recenters the centroid to the average distance of all its datapoints.
        # It returns nothing, but it updates cluster centers 
        
        for row in range(len(self.cluster_centers_)):
            cluster_indices = np.where(self.labels_ == row)[0]    

    def group_clusters(self, input: np.ndarray) -> None:
        for row in range(np.shape(input)[0]):
                bestChoice = self.calculate_distance(np.array(input[row,:]),self.cluster_centers_[0])
                for c in range(self.n_clusters):
                    calculatedDistance = self.calculate_distance(np.array(input[row,:]),self.cluster_centers_[c])
                    if calculatedDistance < bestChoice: #if it is smaller, then update this to be the new best choice
                        bestChoice = calculatedDistance
                    if calculatedDistance < bestChoice:
                        self.labels_[row] = c #label index changes to become cluster index, which updates them to be with the cluster