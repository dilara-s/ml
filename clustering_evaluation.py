import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(X):
    inertia = []
    silhouette_scores = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    optimal_k = K_range[np.argmax(silhouette_scores)]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(K_range, inertia, marker='o', linestyle='-', color='b', label='Inertia')
    ax1.set_xlabel('Количество кластеров')
    ax1.set_ylabel('Инерция', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(K_range, silhouette_scores, marker='s', linestyle='-', color='r', label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Оптимальное количество кластеров: метод локтя и силуэт')
    plt.show()

    return optimal_k