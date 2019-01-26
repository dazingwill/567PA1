import numpy as np
from typing import List
from hw1_knn import KNN

import math
from collections import Counter


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    branch_counts = [sum(branch) for branch in branches]
    all_count = sum(branch_counts)
    after_entropy = sum(
        -sum((count / branch_count * math.log2(count / branch_count)) if count > 0 else 0 for count in branch) * branch_count / all_count
        for branch, branch_count in zip(branches, branch_counts)
    )
    return S - after_entropy


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    prune_node, improve = decisionTree.root_node.pick_prune_node(X_test, y_test)
    while prune_node:
        prune_node.splittable = False
        prune_node, improve = decisionTree.root_node.pick_prune_node(X_test, y_test)


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    counter = Counter(zip(real_labels, predicted_labels))
    tp = counter[(1, 1)]
    fp = counter[(0, 1)]
    fn = counter[(1, 0)]
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    res = math.sqrt(sum(
        (a - b) * (a - b) for a, b in zip(point1, point2)
    ))
    return res


#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return sum(a * b for a, b in zip(point1, point2))


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    return -math.exp(-0.5 * sum((a - b) * (a - b) for a, b in zip(point1, point2)))


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    numerator = sum(a * b for a, b in zip(point1, point2))
    denominator = math.sqrt(sum(a * a for a in point1)) * math.sqrt(sum(a * a for a in point2))
    return 1 - numerator / denominator


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    max_k = min(30, len(Xtrain))

    best_k = 0
    best_model = None
    best_f1 = -99999.9
    best_func = None
    for distance_func_name in ('euclidean', 'gaussian', 'inner_prod', 'cosine_dist'):
        for k in range(1, max_k, 2):
            model = KNN(k, distance_funcs[distance_func_name])
            model.train(Xtrain, ytrain)

            valid_f1_score = f1_score(yval, model.predict(Xval))
            # print('[part 1.1] {name}\tk: {k:d}\t'.format(name=distance_func_name, k=k) +
            #       'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

            if best_f1 < valid_f1_score:
                best_f1 = valid_f1_score
                best_model = model
                best_k = k
                best_func = distance_func_name

    # print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=best_func, best_k=best_k) +
    #       'valid f1 score: {valid_f1_score:.5f}'.format(valid_f1_score=best_f1))
    return best_model, best_k, best_func


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    max_k = min(30, len(Xtrain))

    best_k = 0
    best_model = None
    best_f1 = -99999.9
    best_func = None
    best_scaler = None

    for scaling_class_name in ('min_max_scale', 'normalize'):
        scaling = scaling_classes[scaling_class_name]()
        tmpXtrain = scaling(Xtrain)
        tmpXval = scaling(Xval)
        for distance_func_name in ('euclidean', 'gaussian', 'inner_prod', 'cosine_dist'):
            for k in range(1, max_k, 2):
                model = KNN(k, distance_funcs[distance_func_name])
                model.train(tmpXtrain, ytrain)

                valid_f1_score = f1_score(yval, model.predict(tmpXval))
                # print('[part 1.2] {name}\tscaler: {scaler}\tk: {k:d}\t'
                #       .format(name=distance_func_name, scaler=scaling_class_name, k=k) +
                #       'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

                if best_f1 < valid_f1_score:
                    best_f1 = valid_f1_score
                    best_model = model
                    best_k = k
                    best_func = distance_func_name
                    best_scaler = scaling_class_name

    # print('[part 1.2] {name}\tscaler: {scaler}\tbest_k: {best_k:d}\t'
    #       .format(name=best_func, scaler=best_scaler, best_k=best_k) +
    #       'valid f1 score: {valid_f1_score:.5f}'.format(valid_f1_score=best_f1))

    return best_model, best_k, best_func, best_scaler


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        res = []
        for feature in features:
            denominator = math.sqrt(sum(a * a for a in feature))
            if denominator == 0:
                denominator = 1.0
            res.append([item / denominator for item in feature])
        return res


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.mins = None
        self.scales = None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        if not self.mins:
            self.mins = []
            self.scales = []
            for items in zip(*features):
                maxv = max(items)
                minv = min(items)
                self.mins.append(minv)
                self.scales.append(maxv - minv if maxv != minv else len(features[0]))

        res = []
        for feature in features:
            res.append([(item - self.mins[i]) / self.scales[i] for i, item in enumerate(feature)])
        return res

