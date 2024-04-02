import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, is_leaf=True, prediction=None, split_feature=None, split_value=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = None
        self.right = None

def hoeffding_bound(R, n, delta):
    return np.sqrt((R**2 * np.log(1/delta)) / (2 * n))

class HoeffdingTree:
    def __init__(self, delta=0.05, min_samples_split=2):
        self.root = Node()
        self.delta = delta
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        # Start growing the tree using the training data
        self._grow_tree(self.root, X, y)

    def _grow_tree(self, node, X, y):
        # Check base cases
        if len(X) < self.min_samples_split:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return

        # Calculate current Gini index
        current_gini = self._gini(y)

        # Initialize best split parameters
        best_gini_gain = 0
        best_feature = None
        best_value = None

        # Try splitting on each feature
        for feature in X.columns:
            values = X[feature].unique()
            for value in values:
                left_mask = X[feature] <= value
                right_mask = ~left_mask
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                n_left = sum(left_mask)
                n_right = sum(right_mask)

                # Weighted average Gini index after split
                gini_split = (n_left * left_gini + n_right * right_gini) / len(X)
                gini_gain = current_gini - gini_split

                # Check if this is the best split so far and if it passes the Hoeffding bound
                if gini_gain > best_gini_gain and gini_gain > hoeffding_bound(1, len(X), self.delta):
                    best_gini_gain = gini_gain
                    best_feature = feature
                    best_value = value
            print("Tree is growing")

        # If a split was found that satisfies the Hoeffding bound
        if best_feature is not None:
            node.is_leaf = False
            node.split_feature = best_feature
            node.split_value = best_value
            left_mask = X[best_feature] <= best_value
            right_mask = ~left_mask
            node.left = Node()
            node.right = Node()
            self._grow_tree(node.left, X[left_mask], y[left_mask])
            self._grow_tree(node.right, X[right_mask], y[right_mask])
        else:
            # No valid split was found, make this a leaf node
            node.is_leaf = True
            node.prediction = self._majority_class(y)

    def _gini(self, y):
        # Calculate Gini impurity for a set of labels
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum() # [P(y=1), P(y=2)]
        return 1 - np.sum(probabilities ** 2)

    def _majority_class(self, y):
        # Determine the majority class for a set of labels
        return y.mode()[0]

    def predict(self, X):
        # Make predictions for each sample
        return np.array([self._predict_sample(self.root, sample) for _, sample in X.iterrows()])

    def _predict_sample(self, node, sample):
        # Recursive function to predict the class for a single sample
        if node.is_leaf:
            return node.prediction
        else:
            if sample[node.split_feature] <= node.split_value:
                return self._predict_sample(node.left, sample)
            else:
                return self._predict_sample(node.right, sample)

def main():
    print("HAT Implementation: ")
    file_path = 'Skin_NonSkin 2.txt'
    data = np.loadtxt(file_path, delimiter='\t')

    # Split the data into features and target variable
    X = data[:, :-1].astype(int) # Going to be an array of 245057x3
    y = data[:, -1].astype(int) # Going to be an array of 245057x1

    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)

    tree = HoeffdingTree() #Create Hoeffding Tree
    tree.fit(X_df, y_df) # Fit the tree to the data
    predictions = tree.predict(X_df)
    actual_labels = y_df

    accuracy = accuracy_score(actual_labels, predictions)

    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()
