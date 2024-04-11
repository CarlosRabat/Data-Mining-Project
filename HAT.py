import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from Node import Node

# Hoeffding Adaptive Tree Implementation (This will be used to learn how to build the HAT and then the EFHAT)
class HoeffdingAdaptiveTree:
    # Initialize
    def __init__(self, delta=0.05, min_samples_split=2):
        self.root = Node()
        self.delta = delta
        self.min_samples_split = min_samples_split
        self.alternateTrees = {}
        self.currentTree = None
        self.seenSamples = 0

    # Create the tree
    def fit(self, X, y):
        self.createRoot(self.root, X, y)

    # Keep creating the tree
    def createRoot(self, node, X, y):
        self.seenSamples += 1
        # Calculate Gini score/impurity
        current_gini = self.gini(y) 

        # Variables for best split feature
        best_gini_gain = 0
        best_feature = None
        best_value = None
        column = 0

        # Try splitting on each feature
        # For each unique feature in each column, find the best feature to split on and assign it to best_feature variable.
        for feature in X.columns:
            column += 1
            print("Finding features in column: " + str(column))
            values = X[feature].unique() # Get unique features so the code doesn't try the same feature multiple times
            for value in values:
                left_mask = X[feature] <= value # All features less than value
                right_mask = ~left_mask # All features not in left_mask
                left_gini = self.gini(y[left_mask]) # Calculate gini score of left_mask
                right_gini = self.gini(y[right_mask]) # Calculate gini score of right_mask
                n_left = sum(left_mask) # Number of instances in left_mask
                n_right = sum(right_mask) # Number of instances in right_mask

                # Weighted average Gini index after split
                gini_split = (n_left * left_gini + n_right * right_gini) / len(X)
                gini_gain = current_gini - gini_split # How much has been gained for splitting on this value

                # Check for best feature to split on
                if gini_gain > best_gini_gain and gini_gain > self.hoeffding_bound(1, len(X), self.delta):
                    best_gini_gain = gini_gain
                    best_feature = feature
                    best_value = value
            
        column = 0
        # If best_feature is found, split on this feature. If not, make it a leaf.
        if best_feature is not None:
            node.is_leaf = False
            node.split_feature = best_feature
            node.split_value = best_value
            left_mask = X[best_feature] <= best_value
            right_mask = ~left_mask
            node.left = Node()
            node.right = Node()
            self.growTree(X[left_mask], y[left_mask], node.left)
            self.growTree(X[right_mask], y[right_mask], node.right)
        else:
            print("ERROR")
            exit()

    # From this node, start building HTs and updating the tree
    def growTree(self, X, y, node):
        for index, row in X.iterrows():
            label = y.iloc[index].iloc[0].item()
            #print(row[0], row[1], row[2], label)

            best_feature, best_value, best_score = self._evaluate_splits(X, y)
            if best_feature is not None and self._is_significant_split(best_score):
                node.is_leaf = False
                node.split_feature = best_feature
                node.split_value = best_value

                left_mask = X[best_feature] <= best_value
                right_mask = ~left_mask

                node.left = Node()
                node.right = Node()

                # Recursively grow the left and right subtrees
                self.growTree(X[left_mask], y[left_mask], node.left)
                self.growTree(X[right_mask], y[right_mask], node.right)
            else:
                node.is_leaf = True
                node.prediction = self._majority_class(y)

            # ADWIN update and drift check
            old_prediction = node.prediction
            node.adwin.update(y == old_prediction)  # Using accuracy as feedback; adapt as needed
            
            if node.adwin.drift_detected:
                node.reset_to_leaf()

    # Get gini score for the root
    def gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum() # [P(y=1), P(y=2)] for our first dataset
        return 1 - np.sum(probabilities ** 2)

    # Most common label
    def _majority_class(self, y):
        return y.mode()[0].iloc[0]

    # Predict each sample's label
    def predict(self, X):
        return np.array([self._predict_sample(self.root, sample) for _, sample in X.iterrows()])

    # Actual function that predicts each sample's label
    def _predict_sample(self, node, sample):
        if node.is_leaf:
            return node.prediction
        else:
            if sample[node.split_feature] <= node.split_value:
                return self._predict_sample(node.left, sample)
            else:
                return self._predict_sample(node.right, sample)
            
    # Hoeffding bound is used to make decisions about splitting.
    def hoeffding_bound(self, R, n, delta):
        return np.sqrt((R**2 * np.log(1/delta)) / (2 * n))

