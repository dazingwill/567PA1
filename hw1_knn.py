from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

from collections import Counter

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function
        self.train_features = None
        self.train_labels = None

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.train_features = features
        self.train_labels = labels

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        res = []
        for feature in features:
            k_neighbors = self.get_k_neighbors(feature)
            counter = Counter(k_neighbors)
            res.append(counter.most_common(1)[0][0])
        return res

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        distances = (self.distance_function(a, point) for a in self.train_features)
        labels = sorted(zip(distances, self.train_labels))[:self.k]
        labels = [label[1] for label in labels]
        return labels


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
