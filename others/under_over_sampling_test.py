# -*- coding: utf-8 -*-
"""
@author: ytgw
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn import svm

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


def visualize(X, y, centers, n_classes, title):
    """
    データ、SVMでのクラス領域、クラス中心点をグラフ化する
    """
    model = svm.SVC()
    model.fit(X,y)

    # raw data
    markers = ("x", "x", "x")
    colors = ("r", "g", "b")
    for i in range(n_classes):
        X_scatter = X[y==i]
        plt.scatter(X_scatter[:,0], X_scatter[:,1], marker=markers[i], c=colors[i], alpha=0.2)

    # classifier
    x0_min, x0_max = X[:,0].min()-1, X[:,0].max()+1
    x1_min, x1_max = X[:,1].min()-1, X[:,1].max()+1
    x0_mesh, x1_mesh = np.meshgrid(np.arange(x0_min, x0_max, 0.1),
                                   np.arange(x1_min, x1_max, 0.1))
    z = model.predict(np.array([x0_mesh.ravel(), x1_mesh.ravel()]).T)
    z = z.reshape(x1_mesh.shape)
    plt.contourf(x0_mesh, x1_mesh, z, alpha=0.3, cmap=ListedColormap(colors))

    # center
    for i in range(n_classes):
        plt.scatter(centers[i,0], centers[i,1], marker="o", color="w", s=200, edgecolors="k")
    plt.title(title)
    plt.show()

# ----------------------------------------------------------------------
# Make Dataset
# ----------------------------------------------------------------------
centers = np.array([[ 1,  1],
                    [ 1, -1],
                    [-1,  1]])
n_classes = len(centers)
X_original, y_original = make_blobs(n_samples=5000,
                                    n_features=centers.shape[1],
                                    centers=centers,
                                    cluster_std=1,
                                    center_box=(-10,10),
                                    shuffle=False,
                                    random_state=0)

ratio = {0:980, 1:10, 2:10}
X,y = make_imbalance(X_original,
                     y_original,
                     ratio=ratio,
                     random_state=0)

# ----------------------------------------------------------------------
# Original Data
# ----------------------------------------------------------------------
visualize(X_original, y_original, centers, n_classes, title="Original Data")


# ----------------------------------------------------------------------
# Imbalanced Data
# ----------------------------------------------------------------------
visualize(X, y, centers, n_classes, title="Imbalanced Data")


# ----------------------------------------------------------------------
# Under sampling
# ----------------------------------------------------------------------
class_num = [np.sum(y==c) for c in range(n_classes)]
under_ratio = {key:min(class_num) for key in range(n_classes)}
under_sampler = RandomUnderSampler(ratio=under_ratio, random_state=0)
X_under, y_under = under_sampler.fit_sample(X, y)
visualize(X_under, y_under, centers, n_classes, title="Under Sampling")


# ----------------------------------------------------------------------
# Over sampling(Random Copy)
# ----------------------------------------------------------------------
over_ratio = {key:max(class_num) for key in range(n_classes)}
over_sampler = RandomOverSampler(ratio=over_ratio, random_state=0)
X_random_over, y_random_over = over_sampler.fit_sample(X, y)
visualize(X_random_over, y_random_over, centers, n_classes, title="Over sampling(Random Copy)")


# ----------------------------------------------------------------------
# Over sampling(SMOTE)
# ----------------------------------------------------------------------
over_sampler = SMOTE(ratio=over_ratio, random_state=0)
X_smote_over, y_smote_over = over_sampler.fit_sample(X, y)
visualize(X_smote_over, y_smote_over, centers, n_classes, title="Over sampling(SMOTE)")


# ----------------------------------------------------------------------
# Over sampling(ADASYN)
# ----------------------------------------------------------------------
over_sampler = ADASYN(ratio=over_ratio, random_state=0)
X_adasyn_over, y_adasyn_over = over_sampler.fit_sample(X, y)
visualize(X_adasyn_over, y_adasyn_over, centers, n_classes, title="Over sampling(ADASYN)")
