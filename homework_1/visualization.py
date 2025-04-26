import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA


def plot_clusters(X, labels, centroids, iteration, save_path):
    pca = PCA(n_components=2)
    X_2D = pca.fit_transform(X)
    centroids_2D = pca.transform(centroids)

    plt.figure(figsize=(6, 5))
    for i in range(len(centroids)):
        plt.scatter(X_2D[labels == i, 0], X_2D[labels == i, 1], label=f'Кластер {i + 1}')
    plt.scatter(centroids_2D[:, 0], centroids_2D[:, 1], marker='x', s=200, c='black', label='Центроиды')

    plt.legend()
    plt.title(f'Итерация {iteration}')
    plt.savefig(save_path)
    plt.close()


def plot_all_projections(X, labels, feature_names):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (i, j) in zip(axes, combinations(range(X.shape[1]), 2)):
        ax.scatter(X[:, i], X[:, j], c=labels, cmap='viridis', edgecolor='k')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_title(f'Проекция: {feature_names[i]} vs {feature_names[j]}')

    plt.tight_layout()
    plt.show()
