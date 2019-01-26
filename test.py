

import numpy as np
from hw1_knn import KNN
from utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance, cosine_sim_distance
from utils import f1_score, model_selection_without_normalization, model_selection_with_transformation
distance_funcs = {
    'euclidean': euclidean_distance,
    'gaussian': gaussian_kernel_distance,
    'inner_prod': inner_product_distance,
    'cosine_dist': cosine_sim_distance,
}
from data import data_processing
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()



best_model, best_k, best_function = model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval)


from utils import NormalizationScaler, MinMaxScaler
scaling_classes = {
    'min_max_scale': MinMaxScaler,
    'normalize': NormalizationScaler,
}
best_model, best_k, best_function, best_scaler = model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval)




#
# import data
# import hw1_dt as decision_tree
# import utils as Utils
# from sklearn.metrics import accuracy_score
# #
# # features, labels = data.sample_decision_tree_data()
# #
# # # build the tree
# # dTree = decision_tree.DecisionTree()
# # dTree.train(features, labels)
# #
# # # print
# # Utils.print_tree(dTree)
# #
# # # data
# # X_test, y_test = data.sample_decision_tree_test()
# #
# # # testing
# # y_est_test = dTree.predict(X_test)
# # test_accu = accuracy_score(y_est_test, y_test)
# # print('test_accu', test_accu)
#
# #load data
# X_train, y_train, X_test, y_test = data.load_decision_tree_data()
#
# # set classifier
# dTree = decision_tree.DecisionTree()
#
# # training
# dTree.train(X_train.tolist(), y_train.tolist())
#
# # # print
# # Utils.print_tree(dTree)
#
#
# import json
# # testing
# y_est_test = dTree.predict(X_test)
# test_accu = accuracy_score(y_est_test, y_test)
# print('test_accu', test_accu)
#
# Utils.reduced_error_prunning(dTree, X_test, y_test)
#
# y_est_test = dTree.predict(X_test)
# test_accu = accuracy_score(y_est_test, y_test)
# print('test_accu', test_accu)