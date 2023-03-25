import os
import os.path as osp
import shutil
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from tqdm import tqdm


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    c2p = [-1]*(n_samples*2-1)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
                c2p[child_idx] = i+n_samples
            else:
                current_count += counts[child_idx - n_samples]
                c2p[child_idx] = i+n_samples

        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs), c2p


def main():
    if osp.exists('images'):
        shutil.rmtree('images')
    os.makedirs('images')

    np.random.seed(0)
    n = 50
    dendrogram_p = 2
    X, _ = make_blobs(n_samples=n, centers=3,
                      cluster_std=0.50, random_state=0)

    model = AgglomerativeClustering(affinity='euclidean',
                                    linkage='ward',
                                    distance_threshold=0,
                                    n_clusters=None)
    model = model.fit(X)

    fig = plt.figure(figsize=(15, 15))
    Z, c2p = plot_dendrogram(model, truncate_mode='level', p=dendrogram_p)
    plt.xlabel(
        "Number of points in node (or index of point if no parenthesis).",
        fontsize=30)
    plt.tick_params(labelsize=25)
    fig.savefig('images/dendrogram.png')
    plt.close(fig)

    leaves = Z['leaves']

    p2e = dict()
    for e, p in enumerate(leaves):
        p2e[p] = e

    i2e = dict()
    for i in range(n):
        p = i
        while p not in leaves:
            p = c2p[p]
        i2e[i] = p2e[p]

    P = []
    Q = []
    for Ik, Dk in zip(Z['icoord'], Z['dcoord']):
        x1, _, _, x4 = Ik
        y1,  y2, _, y4 = Dk
        if y1 == 0:
            P.append(x1)
        if y4 == 0:
            P.append(x4)
        Q.append((y2, x1, x4))

    P.sort()
    x2e = dict()
    e2x = defaultdict(list)
    for e, x in enumerate(P):
        x2e[x] = e
        e2x[e].append(x)

    x2i = defaultdict(list)
    for i in range(n):
        x2i[P[i2e[i]]].append(i)

    colors = matplotlib.colormaps['tab20'].colors

    # sort Q and add dummy tuple
    Q.sort()
    Q = [(-1, -1, -1)] + Q

    for k, (y2, x1, x4) in tqdm(list(enumerate(Q))):
        if k > 0:
            x2 = (x1+x4)/2
            e1 = x2e[x1]
            e2 = x2e[x4]

            for x in e2x[e1]:
                for i in x2i[x]:
                    x2i[(x2, y2)].append(i)

            for x in e2x[e2]:
                e2x[e1].append(x)
                for i in x2i[x]:
                    x2i[x2].append(i)
                    i2e[i] = e1

            x2e[x2] = e1
            e2x[e1].append((x2, y2))

        fig = plt.figure(figsize=(15, 15))
        for i in range(n):
            plt.scatter(X[i, 0], X[i, 1], color=colors[i2e[i]], s=200)
        plt.title(f'thr={y2:.3f}, p={dendrogram_p}, num_clusters={len(Q)-k}',
                  fontsize=30)
        plt.tick_params(labelsize=25)
        fig.savefig(f'images/scatter_{k}.png')
        plt.close(fig)

        fig = plt.figure(figsize=(15, 15))
        for i in range(n):
            plt.scatter(X[i, 0], X[i, 1], color=colors[i2e[i]], s=200)
            plt.text(X[i, 0], X[i, 1]+0.05, s=str(i), ha='center', fontsize=30)
        plt.title(f'thr={y2:.3f}, p={dendrogram_p}, num_clusters={len(Q)-k}',
                  fontsize=30)
        plt.tick_params(labelsize=25)
        fig.savefig(f'images/scatter_{k}_with_text.png')
        plt.close(fig)

    imgs = []
    for k in range(len(Q)):
        img_path = f'images/scatter_{k}.png'
        img = Image.open(img_path)
        imgs.append(img)
    imgs[0].save('images/scatter.gif',
                 save_all=True, append_images=imgs[1:],
                 optimize=True, duration=1500, loop=0)

    imgs = []
    for k in range(len(Q)):
        img_path = f'images/scatter_{k}_with_text.png'
        img = Image.open(img_path)
        imgs.append(img)
    imgs[0].save('images/scatter_with_text.gif',
                 save_all=True, append_images=imgs[1:],
                 optimize=True, duration=1500, loop=0)


if __name__ == '__main__':
    main()
