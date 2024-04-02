import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from river.drift import ADWIN

# This will be run on just our basic dataset. That set has 3 numerical features (ranging from ~65 to 255) and a predicted binary value (1,2).
# This dataset looks like rgb values (not sure though).

# Basic Node Class for building tree.
class Node:
    def __init__(self, is_leaf=True, prediction=None, split_feature=None, split_value=None, height=0):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = None
        self.right = None
        self.up = None
        self.height = height
        self.adwin = ADWIN()


    def update_height(self):
        left_height = self.left.height if self.left else -1
        right_height = self.right.height if self.right else -1
        self.height = 1 + max(left_height, right_height)

    def reset_to_leaf(self):
        # Reset the node to a leaf
        self.is_leaf = True
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        # Re-calculate the prediction based on current data

    def check_drift(self, error):
        self.adwin.update(error)  # Update the ADWIN instance with the current error or value
        if self.adwin.change_detected:
            # Drift detected, reset or adjust the node as necessary
            print("Change detected in data")
            self.adwin.reset()  # Reset ADWIN to monitor for new changes
            self.reset_to_leaf()

# Hoeffding bound is used to make decisions about splitting.
def hoeffding_bound(R, n, delta):
    return np.sqrt((R**2 * np.log(1/delta)) / (2 * n))

# Hoeffding Tree Implementation (This will be used to learn how to build the HAT and then the EFHAT)
class HoeffdingAdaptiveTree:
    # Initialize
    def __init__(self, delta=0.05, min_samples_split=2):
        self.root = Node()
        self.delta = delta
        self.min_samples_split = min_samples_split

    # Create the tree
    def fit(self, X, y):
        self._grow_tree(self.root, X, y)

    # Keep creating the tree
    def _grow_tree(self, node, X, y):
        # Check base case
        if len(X) < self.min_samples_split:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return

        # Calculate Gini score/impurity
        current_gini = self._gini(y) 

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
                left_gini = self._gini(y[left_mask]) # Calculate gini score of left_mask
                right_gini = self._gini(y[right_mask]) # Calculate gini score of right_mask
                n_left = sum(left_mask) # Number of instances in left_mask
                n_right = sum(right_mask) # Number of instances in right_mask

                # Weighted average Gini index after split
                gini_split = (n_left * left_gini + n_right * right_gini) / len(X)
                gini_gain = current_gini - gini_split # How much has been gained for splitting on this value

                # Check for best feature to split on
                if gini_gain > best_gini_gain and gini_gain > hoeffding_bound(1, len(X), self.delta):
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
            node.left.up = node
            node.right = Node()
            node.right.up = node
            temp_node = node
            while temp_node:
                temp_node.update_height()
                temp_node = temp_node.up
            self._grow_tree(node.left, X[left_mask], y[left_mask])
            self._grow_tree(node.right, X[right_mask], y[right_mask])
        else:
            node.is_leaf = True
            node.prediction = self._majority_class(y)

    def _gini(self, y):
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
    
    def toString(self):
        if not self.root:
            return "The tree is empty"

        # Use a queue for BFS traversal
        queue = [(self.root, 0)]
        while queue:
            current, level = queue.pop(0)
            # Print the current node's details
            if current.is_leaf:
                print(f"Level {level} | Leaf: Prediction={current.prediction}, Height={current.height}")
            else:
                print(f"Level {level} | Node: Split Feature={current.split_feature}, Split Value={current.split_value}, Height={current.height}")

            # Add the child nodes to the queue
            if current.left:
                queue.append((current.left, level + 1)) # type: ignore
            if current.right:
                queue.append((current.right, level + 1)) # type: ignore

def main():
    print("HAT Implementation: ")
    file_path = 'Skin_NonSkin 2.txt'
    data = np.loadtxt(file_path, delimiter='\t')

    # Split the data into features and target variable
    X = data[:, :-1].astype(int) # Going to be an array of 245057x3
    y = data[:, -1].astype(int) # Going to be an array of 245057x1

    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)

    tree = HoeffdingAdaptiveTree() #Create Hoeffding Adapative Tree
    tree.fit(X_df, y_df) # Fit the tree to the data
    predictions = tree.predict(X_df)
    actual_labels = y_df
    tree.toString()

    accuracy = accuracy_score(actual_labels, predictions)

    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()
