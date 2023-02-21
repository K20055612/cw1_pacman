# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
from statistics import mode
from sklearn.tree import DecisionTreeClassifier
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
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2):
        # constructor
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    
    # def fit(self, data, target):
    #     return self.create_tree(np.array(data), np.array(target))

    def build_tree(self, data, target, curr_depth=0):
        # recursive function to build the tree 
        X = data
        Y = target
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(data, target, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["data_left"], best_split["target_left"], curr_depth+1)
                # recur right
                right_subtree =  self.build_tree(best_split["data_right"], best_split["target_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, data, target, num_samples, num_features):
        #function to find the best split
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = data[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                data_left, target_left, data_right, target_right = self.split(data, target, feature_index, threshold)
                # check if childs are not null
                if len(data_left)>0 and len(data_right)>0:
                    y, left_y, right_y = target, target_left, target_right
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["data_left"] = data_left
                        best_split["data_right"] = data_right
                        best_split["target_left"] = target_left
                        best_split["target_right"] = target_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, data, target, feature_index, threshold):
        # function to split the data
        data_left, target_left, data_right, target_right = [], [] ,[] ,[]
        for sample_index in range(data.shape[0]):
            if data[sample_index][feature_index]<=threshold:
                data_left.append(data[sample_index])
                target_left.append(target[sample_index])
            else:
                data_right.append(data[sample_index])
                target_right.append(target[sample_index])

        return np.array(data_left), np.array(target_left), np.array(data_right), np.array(target_right), 
    
    def information_gain(self, parent, l_child, r_child, mode="gini"):
        # function to compute information gain
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        return gain
    
    def gini_index(self, y):
        # function to compute gini index
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        # function to compute leaf node
        
        # Y = list(Y)
        # return max(list(Y), key=list(Y).count)
        return mode(list(Y))
    
    def print_tree(self, tree=None, indent=" "):
        # function to print the tree
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, data, target):
        # function to train the tree
        self.root = self.build_tree(np.array(data), np.array(target))
    
    def predict(self, X):
        #function to predict new dataset 
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree=None):
        #function to predict a single data point

        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
