import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Speciation:
    def __init__(self, distance_func):
        self.distance_func = distance_func

    def apply_speciation(self, population):
        raise NotImplementedError("This method should be implemented by subclasses.")


class ThresholdSpeciation(Speciation):
    def __init__(self, similarity_threshold, distance_func):
        super().__init__(distance_func)
        self.similarity_threshold = similarity_threshold

    def apply_speciation(self, population):
        species = []
        for i, ind_i in enumerate(population):
            placed = False
            for spec in species:
                if self.distance_func(ind_i, population[spec[0]]) < self.similarity_threshold:
                    spec.append(i)
                    placed = True
                    break
            if not placed:
                species.append([i])
        return species



class KMeansSpeciation(Speciation):
    def __init__(self, num_clusters, distance_func):
        super().__init__(distance_func)
        self.num_clusters = num_clusters

    def apply_speciation(self, population):
        # Calculate the distance matrix
        distance_matrix = np.array([[self.distance_func(ind_i, ind_j) for ind_j in population] for ind_i in population])
        
        # Use KMeans clustering
        kmeans = KMeans(n_clusters=self.num_clusters)
        labels = kmeans.fit_predict(distance_matrix)
        
        # Group individuals into species based on labels
        species = [[] for _ in range(self.num_clusters)]
        for i, label in enumerate(labels):
            species[label].append(i)
        
        return species


class SilhouetteKMeansSpeciation(Speciation):
    def __init__(self, min_clusters, max_clusters, distance_func):
        super().__init__(distance_func)
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def apply_speciation(self, population):
        # Calculate the distance matrix
        distance_matrix = np.array([[self.distance_func(ind_i, ind_j) for ind_j in population] for ind_i in population])

        best_num_clusters = self.min_clusters
        best_silhouette_score = -1
        best_labels = None

        # Determine the optimal number of clusters using silhouette scores
        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(distance_matrix)
            silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')

            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_num_clusters = n_clusters
                best_labels = labels

        # Group individuals into species based on best_labels
        species = [[] for _ in range(best_num_clusters)]
        for i, label in enumerate(best_labels):
            species[label].append(i)

        return species
