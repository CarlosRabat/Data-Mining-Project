import numpy as np
from sklearn.metrics import accuracy_score

class TreeNode:
    def __init__(self, is_leaf=True, prediction=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.children = {}
        self.stats = None

    def add_child(self, value, node):
        self.children[value] = node

class HAT:

    def predict(root, data_point):
        node = root
        while not node.is_leaf:
            # Traverse the tree based on the data_point's features
            # and the tests at each node. This is a placeholder.
            node = node.children.get(data_point_feature, default_child)
        return node.prediction

    def update_tree(root, data_point, label):
        # This function would update the tree based on the incoming data point.
        # For simplicity, we're not implementing the split logic here.
        pass

def hoeffding_bound(R, n, delta):
    """
    R: Range of the variable. For a binary classifier, it's 1.
    n: Number of observations.
    delta: Confidence level, typically set very low (e.g., 0.05 or 0.01).
    """
    return np.sqrt((R**2 * np.log(1/delta)) / (2 * n))

def main():
    print("HAT Implementation: ")
    file_path = 'Skin_NonSkin 2.txt'
    data = np.loadtxt(file_path, delimiter='\t')

    # Split the data into features and target variable
    X = data[:, :-1]
    y = data[:, -1].astype(int) 

    model = HAT()
    model.fit(X, y)
    predictions = model.predict(X)
    actual_labels = y

    accuracy = accuracy_score(actual_labels, predictions)

    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()