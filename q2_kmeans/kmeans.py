"""Code for HW3 Problem 2: K-Means Clustering for Image Compression."""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OPTS = None

def kmeans(X, k):
    """Run k-means clustering on the provided data.

    Your code should keep doing k-means updates until the cluster assignments stop changing.

    Args:
        - X: Matrix of size (N, D) where the i-th row is the i-th example.
        - k: Number of clusters.

    Returns: A tuple (z, centroids, loss) consisting of:
        - z: Vector of shape (N,) where the i-th entry is the cluster assigned to the i-th example.
        - centroids: Matrix of shape (k, D) where the j-th row is the j-th cluster centroid.
        - loss: The k-means loss of the final assignment
    """
    N, D = X.shape
    z = np.zeros(N)  # Initialize
    loss = None
    # Initialize the centroids to random examples in the dataset
    centroids = X[np.random.choice(np.arange(N), k, replace=False),:]  # k x D matrix

    ### BEGIN_SOLUTION 2a
    raise NotImplementedError
    ### END_SOLUTION 2a
    return (z, centroids, loss)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['basic', 'image'])
    parser.add_argument('--k', '-k', type=int, required=True)
    parser.add_argument('--rng-seed', '-s', type=int, default=0)
    return parser.parse_args()


def main():
    np.random.seed(OPTS.rng_seed)
    if OPTS.mode == 'basic':
        data = []
        with open('data.tsv') as f:
            for line in f:
                x = [float(t) for t in line.strip().split('\t')]
                data.append(x)
        X = np.array(data)
    elif OPTS.mode == 'image':
        image = np.asarray(Image.open('original.png'), dtype='float32')  # H, W, 3
        H, W, _ = image.shape
        X = image[:,:,:3].reshape(-1, 3)

    z, centroids, loss = kmeans(X, OPTS.k)
    print(f'K-means loss: {loss:.4f}')

    if OPTS.mode == 'basic':
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(X[:,0], X[:,1], marker='.', c=list(z), cmap='tab10')
        plt.scatter(centroids[:,0], centroids[:,1], marker='*',
                    s=50, edgecolors='k', c=list(range(OPTS.k)), cmap='tab10')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=['Cluster {}'.format(i) for i in range(OPTS.k)])
        plt.savefig(f'clusters_k{OPTS.k}.png')
    elif OPTS.mode == 'image':
        centroids_int = np.uint8(centroids)
        new_image_np = centroids_int[z,:].reshape(H, W, 3)
        new_image = Image.fromarray(new_image_np)
        new_image.save(f'compressed_k{OPTS.k}.png')


if __name__ == '__main__':
    OPTS = parse_args()
    main()

