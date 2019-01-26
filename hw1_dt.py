from collections import Counter

import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])

        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        features = np.array(self.features)
        record_num, attr_num = features.shape

        label_vals = np.unique(self.labels)
        label2index = {label_val: index for index, label_val in enumerate(label_vals)}

        max_gain = -1
        max_index = -1
        uniq_split = None
        for feature_index in range(attr_num):
            feature = features[:, feature_index]
            feature_vals = list(np.unique(feature))
            if len(feature_vals) < 2:
                continue
            feature_val2index = {feature_val: index for index, feature_val in enumerate(feature_vals)}

            branches = np.zeros((len(feature_vals), len(label_vals)))
            counter = Counter(zip(feature, self.labels))
            for item, label in counter:
                branches[feature_val2index[item]][label2index[label]] = counter.get((item, label))

            gain = Util.Information_Gain(1.0, branches)
            if max_gain < gain or (max_gain == gain and len(uniq_split) < len(feature_vals)):
                max_gain = gain
                max_index = feature_index
                uniq_split = feature_vals

        if max_index == -1:
            self.splittable = False
            return
        self.dim_split = max_index
        self.feature_uniq_split = uniq_split

        splited_features = {feature_val: [] for feature_val in self.feature_uniq_split}
        splited_labels = {feature_val: [] for feature_val in self.feature_uniq_split}
        for record, label in zip(features, self.labels):
            splited_features[record[self.dim_split]].append(record)
            splited_labels[record[self.dim_split]].append(label)

        for feature_val in self.feature_uniq_split:
            child = TreeNode(splited_features[feature_val], splited_labels[feature_val], self.num_cls)
            if child.splittable:
                child.split()
            self.children.append(child)

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if not self.splittable:
            return self.cls_max
        feature_val = feature[self.dim_split]

        try:
            feature_val_index = self.feature_uniq_split.index(feature_val)
        except ValueError:
            return self.cls_max
        child = self.children[feature_val_index]
        return child.predict(feature)

    def pick_prune_node(self, x_valid, y_valid):
        if not self.splittable:
            return None, 0

        # calculate this node
        correct_num = 0
        prune_correct_num = 0
        for feature, label in zip(x_valid, y_valid):
            predicts = self.predict(feature)
            if predicts == label:
                correct_num += 1
            if self.cls_max == label:
                prune_correct_num += 1
        improve = prune_correct_num - correct_num
        res_node = self if improve > 0 else None

        # calculate children
        splited_features = {feature_val: [] for feature_val in self.feature_uniq_split}
        splited_labels = {feature_val: [] for feature_val in self.feature_uniq_split}
        for record, label in zip(x_valid, y_valid):
            feature_val = record[self.dim_split]
            if feature_val in splited_features:
                splited_features[feature_val].append(record)
                splited_labels[feature_val].append(label)

        for feature_val_index, feature_val in enumerate(self.feature_uniq_split):
            child = self.children[feature_val_index]
            child_prune_node, child_improve = child.pick_prune_node(
                splited_features[feature_val], splited_labels[feature_val])
            if child_prune_node and child_improve > improve:
                improve = child_improve
                res_node = child_prune_node
        return res_node, improve

