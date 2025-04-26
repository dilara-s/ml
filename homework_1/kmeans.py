import numpy as np
import random
import imageio
from visualization import plot_clusters


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def initialize_centroids(X, k):
    return X[random.sample(range(len(X)), k)]


def assign_clusters(X, centroids):
    clusters = [[] for _ in range(len(centroids))]
    labels = np.zeros(len(X), dtype=int)
    for i, point in enumerate(X):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx].append(point)
        labels[i] = cluster_idx
    return clusters, labels


def update_centroids(clusters):
    return np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else random.choice(X) for cluster in clusters])


def k_means(X, k, max_iters=10, tol=1e-4):
    centroids = initialize_centroids(X, k)
    prev_centroids = np.zeros_like(centroids)
    iteration = 0
    images = []

    while iteration < max_iters and np.linalg.norm(centroids - prev_centroids) > tol:
        clusters, labels = assign_clusters(X, centroids)
        prev_centroids = centroids.copy()
        centroids = update_centroids(clusters)

        img_path = f'kmeans_iter_{iteration}.png'
        plot_clusters(X, np.array(labels), centroids, iteration, img_path)
        images.append(imageio.v2.imread(img_path))

        iteration += 1

    imageio.mimsave('kmeans_animation.gif', images, duration=1)
    return labels