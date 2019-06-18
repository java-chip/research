#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:08:01 2019

@author: project-dl
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import scipy as sp
#from sklearn.datasets import fetch_openml
from keras.datasets import mnist
from keras.datasets import cifar10
import sklearn.base
import bhtsne
import matplotlib.pyplot as plot
import numpy as np
import argparse


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed

    def fit_transform(self, X):
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.dimensions,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=self.rand_seed
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist', help='mnist, cifar10')
    args = parser.parse_args()
    if args.data == "mnist":
        print("load mnist...")
    #    mnist_data, mnist_labels = fetch_openml('mnist_784', version=1, return_X_y=True)
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        bh_tsne = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1)
        # extract data whose class is 1 or 7
    #    X_train = X_train[(y_train==1) | (y_train==7)]
    #    y_train = y_train[[(y_train == 1) | (y_train == 7)]]
        X_train = np.reshape(X_train, (-1, 784))
    elif args.data == "cifar10":
        print("load cifar10...")
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        bh_tsne = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1)
        X_train = np.reshape(X_train, (-1, 32*32*3))
    tsne = bh_tsne.fit_transform(X_train)
    xmin = tsne[:,0].min()
    xmax = tsne[:,0].max()
    ymin = tsne[:,1].min()
    ymax = tsne[:,1].max()

    plot.figure( figsize=(16,12) )
    for _y,_label in zip(tsne[::20],y_train[::20]):
        if _label == 0:
            plot.text(_y[0], _y[1], _label, color='b')
        elif _label == 1:
            plot.text(_y[0], _y[1], _label, color='r')
        elif _label == 2:
            plot.text(_y[0], _y[1], _label, color='violet')
        elif _label == 3:
            plot.text(_y[0], _y[1], _label, color='orange')
        elif _label == 4:
            plot.text(_y[0], _y[1], _label, color='gray')
        elif _label == 5:
            plot.text(_y[0], _y[1], _label, color='black')
        elif _label == 6:
            plot.text(_y[0], _y[1], _label, color='cyan')
        elif _label == 7:
            plot.text(_y[0], _y[1], _label, color='g')
        elif _label == 8:
            plot.text(_y[0], _y[1], _label, color='yellow')
        elif _label == 9:
            plot.text(_y[0], _y[1], _label, color='lime')

    plot.axis([xmin,xmax,ymin,ymax])
    plot.xlabel("component 0")
    plot.ylabel("component 1")
    if args.data == "mnist":
        plot.title("MNIST t-SNE visualization")
    elif args.data == "cifar10":
        plot.title("cifar10 t-SNE visualization")
    plot.savefig("mnist_tsne.png")