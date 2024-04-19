# %% [markdown]
# # Extremely Fast Decision Tree Implementation

# %%
import numpy as np


class Node:
    def __init__(self, is_leaf=True, prediction=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_feature = None
        self.split_value = None
        self.children = {}
        self.gain = 0
        self.class_counts = np.zeros(2)
        self.data_batch = []
        self.label_batch = []
        self.counter = 0


def hoeffding_bound(R, n):
    return np.sqrt((R**2 * np.log(1 / 0.10)) / (2 * n))


def entropy(labels):
    # Count the occurrences of each class
    label_counts = np.bincount(labels, minlength=2)

    # Calculate the probabilities for each class
    probabilities = label_counts / np.sum(label_counts)

    # Remove probabilities equal to 0 for log2 calculation
    probabilities = probabilities[probabilities > 0]

    # Calculate the entropy based on its formula
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def information_gain(parent_labels, left_labels, right_labels):

    # Entropy before the split
    entropy_before = entropy(parent_labels)

    # Weighted entropy after the split
    total_size = len(parent_labels)
    left_size = len(left_labels)
    right_size = len(right_labels)

    weighted_entropy = (left_size / total_size) * entropy(left_labels) + (
        right_size / total_size
    ) * entropy(right_labels)

    # Information gain is the reduction in entropy
    return entropy_before - weighted_entropy


def best_split(data, labels):
    features = data.shape[1]
    best_info = {"feature": None, "value": None, "info_gain": -np.inf}
    second_best_info = {"feature": None, "value": None, "info_gain": -np.inf}

    # Iterate though all the features
    for feature in range(features):
        values = np.sort(np.unique(data[:, feature]))
        values_n = len(values) - 1
        # print(values_n)

        # Sort through the unique values
        for i in range(values_n):
            split_value = (values[i] + values[i + 1]) / 2

            # MArk the values with lower than the split
            smaller_values = data[:, feature] <= split_value
            # Negation fo the left
            bigger_values = ~smaller_values

            # Calculate Information Gain
            info_gain = information_gain(
                labels, labels[smaller_values], labels[bigger_values]
            )

            if info_gain > best_info["info_gain"]:
                best_info, second_best_info = {
                    "feature": feature,
                    "value": split_value,
                    "info_gain": info_gain,
                }, best_info

    return best_info, second_best_info


class EFDT:
    def __init__(self, batch_size=50):
        self.root = Node(is_leaf=True, prediction=0)
        self.batch_size = batch_size

    def _fit_single(self, x, y):
        node = self.root
        while not node.is_leaf:
            if x[node.split_feature] < node.split_value:
                node = node.children["left"]
            elif x[node.split_feature] > node.split_value:
                node = node.children["right"]
            else:
                # print(node.split_feature)
                # print(x[node.split_feature])
                # print(node.split_value)
                node.counter += 1
                node.data_batch.append(x)
                node.label_batch.append(y)
                if len(node.data_batch) >= self.batch_size and node.counter > 0:
                    self.reevaluate_best_split(node)
                    node.data_batch = []
                    node.label_batch = []

        node.data_batch.append(x)
        node.label_batch.append(y)
        node.class_counts[y] += 1
        node.prediction = np.argmax(node.class_counts)

        # Splitting
        if node.is_leaf and len(node.data_batch) >= self.batch_size:
            self._attempt_to_split(node)
            node.data_batch = []
            node.label_batch = []

    def _attempt_to_split(self, node):
        X_sub = np.array(node.data_batch)
        y_sub = np.array(node.label_batch)

        if node.class_counts[0] > 0 and node.class_counts[1] > 0:
            best_info, second_best_info = best_split(X_sub, y_sub)

            if best_info["feature"] is not None:

                n = np.sum(node.class_counts)
                epsilon = hoeffding_bound(1, n)

                if (
                    best_info["info_gain"] - 0 > epsilon
                ):  # G(X) is the info gain from an attribute X picked beforehand
                    node.gain = best_info["info_gain"]
                    node.is_leaf = False
                    node.split_feature = best_info["feature"]
                    node.split_value = best_info["value"]
                    node.children["left"] = Node(
                        is_leaf=True, prediction=np.argmax(node.class_counts)
                    )
                    node.children["right"] = Node(
                        is_leaf=True, prediction=np.argmax(node.class_counts)
                    )

        node.data_batch = []
        node.label_batch = []

    def reevaluate_best_split(self, node):
        X_sub = np.array(node.data_batch)
        y_sub = np.array(node.label_batch)

        best_info, second_best_info = best_split(X_sub, y_sub)
        if best_info["feature"] is not None:

            n = np.sum(node.class_counts)
            epsilon = hoeffding_bound(1, n)

            if best_info["info_gain"] - node.gain > epsilon and (
                best_info["feature"] != node.split_feature
                or best_info["value"] != node.split_value
            ):
                node.is_leaf = False
                node.split_feature = best_info["feature"]
                node.split_value = best_info["value"]
                node.children["left"] = Node(
                    is_leaf=True, prediction=np.argmax(node.class_counts)
                )
                node.children["right"] = Node(
                    is_leaf=True, prediction=np.argmax(node.class_counts)
                )

        node.data_batch = []
        node.label_batch = []

    def predict(self, x):
        node = self.root
        while not node.is_leaf:
            if x[node.split_feature] <= node.split_value:
                node = node.children["left"]
            else:
                node = node.children["right"]

        pred = node.prediction
        return pred


# %% [markdown]
# ## Load the Dataset
# Here we are loading the data from the txt file and adding drift to the dataset itself. This means that the distribution changed in the dataset, to achieve this changed in distribution the labels were flipped.

# %%
import numpy as np

file_path = "Skin_NonSkin 2.txt"

# Load the data
data = np.loadtxt(file_path, delimiter="\t")

# Split the data into features and target variable
X = data[:, :-1]
y = data[:, -1].astype(int)
y = y - 1

# %% [markdown]
# ## Evaluate the Results

# %%
model = EFDT()

# Initialize variables
errors_count = 0
error_rates = []

for idx in range(len(y)):
    pred = model.predict(X[idx])

    if pred != y[idx]:
        errors_count += 1

    # Calculate Error Rate
    if idx > 10000:
        error_rate = errors_count / (idx + 1)
        error_rates.append([idx, error_rate])

    # Print Every 10000 Iterations
    if idx % 10000 == 0:
        print(f"Instance: {idx}")

    model._fit_single(X[idx], y[idx])

# %% [markdown]
# # Display the Results

# %%
import matplotlib.pyplot as plt

# Splitting the list into two lists, idxs and errors
idxs, errors = zip(*error_rates)

# Plotting
plt.figure(figsize=(10, 6))  # Optional: Specifies the figure size
plt.plot(
    idxs, errors, marker="o", linestyle="-", color="b"
)  # Marker, linestyle, and color are optional
plt.title("Error Rate Over Time")
plt.xlabel("Index (or Time)")
plt.ylabel("Error Rate")
plt.grid(True)
plt.show()
