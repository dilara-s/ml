import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.datasets import load_iris
from kmeans import k_means
from clustering_evaluation import find_optimal_k
from visualization import plot_all_projections

iris = load_iris()
X = iris.data
feature_names = iris.feature_names

optimal_k = find_optimal_k(X)

final_labels = k_means(X, optimal_k)

plot_all_projections(X, final_labels, feature_names)