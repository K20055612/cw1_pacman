# classifier.py
# Junchi Ren, Nicolae-Marian Gartu, Xiaotian Yuan, Zhuoyuan Li/ February 2023
#
# Decision Tree Classifier using gini index to calculate information gain
from statistics import mode
import numpy as np

class Classifier:
    def __init__(self):
        self.model = DecisionTree()

    def reset(self):
        pass
    
    def fit(self, data, target):
        self.model.fit(data, target)

    def predict(self, data, legal=None):
        return self.model.predict([data]) # pass data as 2D array
        

class Node:
    # class to represent decision tree node and appropriate fields
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class Split:
    # class to represent split data and appropriate fields
    def __init__(self, feature_index=None, threshold=None, data_left=None, data_right=None, target_left=None, target_right=None, info_gain=None):
        
        self.feature_index = feature_index
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.target_left = target_left
        self.target_right = target_right
        self.info_gain = info_gain


class DecisionTree:
    # class to represent Decision tree
    def __init__(self):

        self.root = None
    
    def fit(self, data, target):
        self.root = self.create_tree(np.array(data), np.array(target))
    
    def predict(self, samples):
        # predict a set of samples
        
        preditions = [self.predict_single_sample(sample, self.root) for sample in samples]
        return preditions
    
    def predict_single_sample(self, sample, tree=None):

        if tree.value!=None: return tree.value
        feature_val = sample[tree.feature_index]
        # choose between left subtree and right subtree according to the threashold
        if feature_val<=tree.threshold:
            return self.predict_single_sample(sample, tree.left)
        else:
            return self.predict_single_sample(sample, tree.right)
    

    def create_tree(self, features, targets, depth=0):
        sample_count, feature_count = np.shape(features)
        
        best_split = self.get_best_split(features, targets, sample_count, feature_count)
        # if the data was split and info gain is positive then recursively create the tree
        if best_split != None and best_split.info_gain>0:
            left_subtree = self.create_tree(best_split.data_left, best_split.target_left, depth+1)
            right_subtree =  self.create_tree(best_split.data_right, best_split.target_right, depth+1)
            
            return Node(best_split.feature_index, best_split.threshold, 
                        left_subtree, right_subtree, best_split.info_gain)
        
        # if we get here, either the data cannot be split anymore or we have reached the maximum depth of the tree
        leaf_value = self.calculate_value_of_node(targets)
        return Node(value=leaf_value)
    
    def get_best_split(self, data, target, sample_count, feature_count):
        
        max_info_gain = -float("inf")

        best_split = None
        for feature_index in range(feature_count):
            feature_values = data[:, feature_index]
            # get all possible thresholds
            # in our case a feature could either be 0 or 1 so this is not entirely necessary, but in general it is
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                # calculate the split for this threshold
                data_left, target_left, data_right, target_right = self.split(data, target, feature_index, threshold)
                # if the data could be split continue, otherwise None will be returned, hence plurality classification would
                # occur on the parent node
                if len(data_left) != 0 and len(data_right) != 0:
                    info_gain = self.information_gain(target, target_left, target_right)
                    # if this split results in a better info gain update it as best split
                    if info_gain>max_info_gain:
                        best_split = Split(feature_index, threshold, data_left, data_right, target_left, target_right, info_gain)
                        max_info_gain = info_gain
                        
        # return best split
        return best_split
    
    def split(self, data, target, feature_index, threshold):
        # split the data based on feature index and threshold
        data_left, target_left, data_right, target_right = [], [] ,[] ,[]
        for sample_index in range(data.shape[0]):
            if data[sample_index][feature_index]<=threshold:
                data_left.append(data[sample_index])
                target_left.append(target[sample_index])
            else:
                data_right.append(data[sample_index])
                target_right.append(target[sample_index])

        return np.array(data_left), np.array(target_left), np.array(data_right), np.array(target_right), 
    
    def information_gain(self, parent, left_subtree_labels, right_subtree_labels):
        
        left_subtree_weight = len(left_subtree_labels) / len(parent)
        right_subtree_weight = len(right_subtree_labels) / len(parent)


        # formula from slides for info gain
        gain = self.calculate_gini_impurity(parent) - (left_subtree_weight*self.calculate_gini_impurity(left_subtree_labels) + right_subtree_weight*self.calculate_gini_impurity(right_subtree_labels))
        return gain
    
    def calculate_gini_impurity(self, y):
        
        classes = np.unique(y)
        gini = 0
        # formula from slides for gini impurity
        for class_label in classes:
            class_ratio = np.count_nonzero(y == class_label) / len(y)
            gini = gini +  class_ratio**2

        return 1 - gini
        
    def calculate_value_of_node(self, Y):
        # get the predominant class using mode
        return mode(list(Y))
    
    