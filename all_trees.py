import numpy as np
import matplotlib.pyplot as plt
from river.drift import ADWIN

""" 
Script contains a function for the following trees to compare them in the Evaluate_Results notebook.
    EFDT
    HAT
    EFHAT
"""


def EFDT():
    """Builds the Extremely Fast Decision Tree.

    Returns:
        model: EFDT
    """

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
        # We set delta to be 0.1 as an arbitrary number
        # R is the range of random variables
        # Since we have a binary random variable R = 1
        return np.sqrt((R**2 * np.log(1/0.10)) / (2 * n))

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
        
        weighted_entropy = (left_size / total_size) * entropy(left_labels) + \
                            (right_size / total_size) * entropy(right_labels)

        # Information gain is the reduction in entropy
        return entropy_before - weighted_entropy

    def best_split(data, labels):
        features = data.shape[1]
        best_info = {'feature': None, 'value': None, 'info_gain': -np.inf}
        second_best_info = {'feature': None, 'value': None, 'info_gain': -np.inf}
        
        # Iterate though all the features
        for feature in range(features):
            values = np.sort(np.unique(data[:, feature])) 
            values_n = len(values) -1
            #print(values_n)
            
            # Sort through the unique values
            for i in range(values_n):
                split_value = (values[i] + values[i+1]) / 2
                
                # MArk the values with lower than the split
                smaller_values = data[:, feature] <= split_value
                #Negation fo the left
                bigger_values = ~smaller_values
                
                # Calculate Information Gain
                info_gain = information_gain(labels, labels[smaller_values], labels[bigger_values])
                
                if info_gain > best_info['info_gain']:
                    best_info, second_best_info = {
                        'feature': feature,
                        'value': split_value,
                        'info_gain': info_gain
                    }, best_info
                    
        return best_info, second_best_info

    class EFDT:
        def __init__(self, batch_size=50):
            self.root = Node(is_leaf=True, prediction=0)
            self.batch_size = batch_size
        
        def _fit_single(self, x, y):
            node = self.root
            
            # Traverse through the tree until you get to a leaf node
            while not node.is_leaf:
                if x[node.split_feature] < node.split_value:
                    node.counter += 1
                    node.data_batch.append(x)
                    node.label_batch.append(y)
                    #Check if we need to reevaluate the split
                    if len(node.data_batch) >= self.batch_size and node.counter > 0:
                        self.reevaluate_best_split(node)
                        node.data_batch = []
                        node.label_batch = []
                    node = node.children['left']  
                    
                elif x[node.split_feature] > node.split_value:
                    node.counter += 1
                    node.data_batch.append(x)
                    node.label_batch.append(y)
                    #Check if we need to reevaluate the split
                    if len(node.data_batch) >= self.batch_size and node.counter > 0:
                        self.reevaluate_best_split(node)
                        node.data_batch = []
                        node.label_batch = []
                    node = node.children['right']

            # Add the current information to the leaf
            node.data_batch.append(x)
            node.label_batch.append(y)
            node.class_counts[y] += 1
            node.prediction = np.argmax(node.class_counts)

            # Split the Node if the criteria is met
            if node.is_leaf and len(node.data_batch) >= self.batch_size:  
                self._attempt_to_split(node)
        
        def _attempt_to_split(self, node):
            X_sub = np.array(node.data_batch)
            y_sub = np.array(node.label_batch)

            if (node.class_counts[0] > 0 and node.class_counts[1] > 0):
                # Get the best split
                best_info, second_best_info = best_split(X_sub, y_sub)

                if best_info['feature'] is not None:

                    n = np.sum(node.class_counts)
                    epsilon = hoeffding_bound(1, n)
                    
                    # Check to see if the criteria is met
                    if best_info['info_gain'] - 0 > epsilon: 
                        
                        # Update the node info
                        node.gain = best_info['info_gain']
                        node.is_leaf = False
                        node.split_feature = best_info['feature']
                        node.split_value = best_info['value']
                        
                        #Assign the children the current prediction
                        node.children['left'] = Node(is_leaf=True, prediction=np.argmax(node.class_counts))
                        node.children['right'] = Node(is_leaf=True, prediction=np.argmax(node.class_counts))

            node.data_batch = []
            node.label_batch = []

        def reevaluate_best_split(self, node):
            X_sub = np.array(node.data_batch)
            y_sub = np.array(node.label_batch)

            best_info, second_best_info = best_split(X_sub, y_sub)
            if best_info['feature'] is not None:
                n = np.sum(node.class_counts)
                epsilon = hoeffding_bound(1, n)

                # Check to see if the current split value is still the best split value 
                if best_info['info_gain'] - node.gain > epsilon and (best_info['feature'] != node.split_feature or best_info['value'] != node.split_value):
                    print("Reconfiguring")
                    node.is_leaf = False
                    node.split_feature = best_info['feature']
                    node.split_value = best_info['value']
                    
                    # Drop the Rest of the tree
                    node.children['left'] = Node(is_leaf=True, prediction=np.argmax(node.class_counts))
                    node.children['right'] = Node(is_leaf=True, prediction=np.argmax(node.class_counts))

            node.data_batch = []
            node.label_batch = []
        
        def predict(self, x):
            node = self.root
            
            #Iterate through the whole tree
            while not node.is_leaf:
                if x[node.split_feature] <= node.split_value:
                    node = node.children['left']
                else:
                    node = node.children['right']
                    
            # Get the prediction 
            pred = node.prediction
            return pred

    return EFDT()


def HAT():
    """Builds a Hoeffding Adaptive Tree
    Returns:
        model: HAT
    """
    class Node:
        def __init__(self, is_leaf=True, prediction=None, is_alternate=False, parent=None, path=[]):
            self.is_leaf = is_leaf
            self.prediction = prediction
            self.split_feature = None
            self.split_value = None
            self.children = {}
            self.class_counts = np.zeros(2)
            self.adwin = ADWIN()
            self.alternate_tree = None
            self.n_predictions = 0
            self.n_correct_predictions = 0
            self.data_batch = []
            self.label_batch = []
            self.parent = parent
            self.path = tuple(path)
            self.is_alternate = is_alternate
            self.alt_order = 0

        @property
        def accuracy(self):
            if self.n_predictions == 0:
                return 0
            return self.n_correct_predictions / self.n_predictions

    def hoeffding_bound(R, n):
        return np.sqrt((R**2 * np.log(1/0.10)) / (2 * n))

    def entropy(labels):
        # Count the occurrences of each class
        label_counts = np.bincount(labels, minlength=2)
        
        # Calculate the probabilities for each class
        probabilities = label_counts / np.sum(label_counts)
        
        # Remove probabilities equal to 0 for log2 calculation
        probabilities = probabilities[probabilities > 0]
        
        # Calculate the entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy 

    def information_gain(parent_labels, left_labels, right_labels):
        # Entropy before the split
        entropy_before = entropy(parent_labels)
        
        # Weighted entropy after the split
        total_size = len(parent_labels)
        left_size = len(left_labels)
        right_size = len(right_labels)
        
        weighted_entropy = (left_size / total_size) * entropy(left_labels) + \
                            (right_size / total_size) * entropy(right_labels)

        # Information gain is the reduction in entropy
        return entropy_before - weighted_entropy

    def best_split(data, labels):
        features = data.shape[1]
        best_info = {'feature': None, 'value': None, 'info_gain': -np.inf}
        second_best_info = {'feature': None, 'value': None, 'info_gain': -np.inf}
        
        # Iterate though all the features
        for feature in range(features):
            values = np.sort(np.unique(data[:, feature]))
            values_n = len(values) - 1
            
            # Sort through the unique values
            for i in range(values_n):
                split_value = (values[i] + values[i+1]) / 2
                
                # Mark the values with lower than the split
                smaller_values = data[:, feature] <= split_value
                #Negation fo the left
                bigger_values = ~smaller_values
                
                # Calculate Information Gain
                info_gain = information_gain(labels, labels[smaller_values], labels[bigger_values])
                if info_gain > best_info['info_gain']:
                    best_info, second_best_info = {
                        'feature': feature,
                        'value': split_value,
                        'info_gain': info_gain
                    }, best_info
                    
        return best_info, second_best_info

    class HAT:
        def __init__(self, batch_size=50):
            self.root = Node(is_leaf=True, prediction=0)
            self.alt_trees = 0
            self.pruned_trees = 0
            self.batch_size = batch_size

        def HATGrow(self, x, y, node=None):
            # If node is None, start from root
            if node == None:
                node = self.root
            # While node is not a leaf, traverse the tree and check for alternate trees. If there is an alternate tree, pass it the data stream and grow it as well
            while not node.is_leaf:
                if x[node.split_feature] <= node.split_value:
                    if node.alternate_tree:
                        self.HATGrow(x, y, node.alternate_tree)
                    if node.is_leaf:
                        break
                    node = node.children['left']
                else:
                    if node.alternate_tree:
                        self.HATGrow(x, y, node.alternate_tree)
                    if node.is_leaf:
                        break
                    node = node.children['right']

            # Update the data_batch and label_batch of the node for best_split calculations later
            node.data_batch.append(x)
            node.label_batch.append(y)

            # Update that class counts and prediction of the leaf node
            correct = False
            node.class_counts[y] += 1
            node.prediction = np.argmax(node.class_counts)
            if node.prediction == y:
                correct = True
            node.n_predictions = np.sum(node.class_counts)
            node.n_correct_predictions = np.max(node.class_counts)

            # Update the accuracy of each node from the root to the leaf it got classified at
            self.update_whole_tree_accuracy(node, correct, y)

            # Split the node if the data_batch size is reached
            if len(node.data_batch) >= self.batch_size:  
                self._attempt_to_split(node)
            
            if node.parent:
                # For each node in the nodes path, update the estimators (ADWIN)
                for nodeX in node.path:
                    nodeX.adwin.update(1 if y != nodeX.prediction else 0)
                    # If drift is detected, create an alternate tree if there is not already one for that node and the node isnt an alternate itself.
                    if nodeX.adwin.drift_detected:
                        # If there is no alternate tree and the node is not a part of the alternate tree itself, create a alternate tree and recursively grow it.
                        if nodeX.alternate_tree is None and not node.is_alternate:
                            self.alt_trees += 1
                            node.alt_order = self.alt_trees
                            nodeX.alternate_tree = Node(is_leaf=True, prediction=np.argmax(nodeX.class_counts), is_alternate=True, parent=nodeX.parent)
                            self.HATGrow(x, y, nodeX.alternate_tree)
                            continue
                    
                    self.possibly_replace(nodeX)
            else:
                assert node == self.root or node == self.root.alternate_tree, "ERROR"
                node.adwin.update(1 if y != node.prediction else 0)
                if node.adwin.drift_detected:
                    # If there is no alternate tree and the node is not a part of the alternate tree itself, create a alternate tree and recursively grow it.
                    if node.alternate_tree is None and not node.is_alternate:
                        self.alt_trees += 1
                        node.alt_order = self.alt_trees
                        node.alternate_tree = Node(is_leaf=True, prediction=np.argmax(node.class_counts), is_alternate=True, parent=node.parent)
                        self.HATGrow(x, y, node.alternate_tree)
                self.possibly_replace(node)
        
        def possibly_replace(self, node):
            if node.alternate_tree:
                main_tree_accuracy = node.accuracy
                alt_tree_accuracy = node.alternate_tree.accuracy
                if alt_tree_accuracy > main_tree_accuracy:
                    self.pruned_trees += 1
                    node.is_leaf = node.alternate_tree.is_leaf
                    node.split_feature = node.alternate_tree.split_feature
                    node.split_value = node.alternate_tree.split_value
                    node.children = node.alternate_tree.children
                    node.n_correct_predictions = node.alternate_tree.n_correct_predictions
                    node.n_predictions = node.alternate_tree.n_predictions
                    print(f"Alternate Tree Replacing Old Tree at depth {len(node.path)}")
                    node.alternate_tree = None

        def update_is_alternate_status(self, children):
            for child in children.values():
                child.is_alternate = False
                self.update_is_alternate_status(child.children)

        # This method updates the accuracy of each node in the path that a piece of data followed down.
        # If the piece of data is classified correctly, the accuracy increases.
        def update_whole_tree_accuracy(self, node, correct, y):
            for x in node.path:
                x.class_counts[y] += 1
                x.prediction = np.argmax(x.class_counts)
                x.n_predictions += 1
                if correct:
                    x.n_correct_predictions += 1

        # Attempt to split the node
        def _attempt_to_split(self, node):
            X_sub = np.array(node.data_batch)
            y_sub = np.array(node.label_batch)
            
            best_info, second_best_info = best_split(X_sub, y_sub)
            if best_info['feature'] is not None: 
                n = np.sum(node.class_counts)
                epsilon = hoeffding_bound(1, n)
                
                # Check if the split is significant
                if best_info['info_gain'] - second_best_info['info_gain'] > epsilon:
                    newPath = list(node.path) + [node]
                    node.is_leaf = False
                    node.split_feature = best_info['feature']
                    node.split_value = best_info['value']
                    # Set leaf to True, inherit parent's prediction, set parent to the node, have leaf inherit parents alternate tree status
                    # give children node's path plus itself
                    node.children['left'] = Node(is_leaf=True, prediction=np.argmax(node.class_counts), parent=node,
                                                is_alternate=node.is_alternate, path=newPath)
                    node.children['right'] = Node(is_leaf=True, prediction=np.argmax(node.class_counts), parent=node, 
                                                is_alternate=node.is_alternate, path=newPath)
                    # The nodes accuracy is set to 0, and the accuracy of all nodes above it must be recalculated.
                    # This makes it a true accuracy calculation with a datastream because the accuracy of the tree and node will change with
                    # splits and classifying a new piece of data
                    self.split_accuracy_recalc(node)

            node.data_batch = []
            node.label_batch = []

        # Recalculate the accuracy of each node in the tree based after splitting a node
        def split_accuracy_recalc(self, node):
            predictions = node.n_predictions
            correct_prediction = node.n_correct_predictions
            for x in node.path:
                x.n_predictions -= predictions
                x.n_correct_predictions -= correct_prediction
            node.n_predictions = 0
            node.n_correct_predictions = 0
                
        # Predict where the data is going
        def predict(self, x):
            node = self.root
            while not node.is_leaf:
                if x[node.split_feature] <= node.split_value:
                    node = node.children['left']
                else:
                    node = node.children['right']

            pred = node.prediction
            return pred
        
        def toString(self, node=None, depth=0):
            if node is None:
                node = self.root
            result = "  " * depth + f"Node(Leaf: {node.is_leaf}, Prediction: {node.prediction}, Alternate: {node.is_alternate}, "
            result += f"Split: [{node.split_feature}, {node.split_value}], N_Predictions: {node.n_predictions}, "
            result += f"N_Correct: {node.n_correct_predictions}, Class Counts: {node.class_counts.tolist()})\n"
            
            for child_key, child in node.children.items():
                result += "  " * (depth + 1) + f"{child_key.capitalize()} Child:\n"
                result += self.toString(child, depth + 2)
            
            return result

        def save_tree_to_file(self, filename='HAT_structure.txt'):
            tree_structure = self.toString()
            with open(filename, 'w') as file:
                file.write(tree_structure)
            print(f"Tree structure saved to {filename}")

        def cleanup(self, node=None):
            if node is None:
                node = self.root
            
            if node.alternate_tree and node.alternate_tree.accuracy <= node.accuracy:
                print("Pruning alternate tree")
                self.pruned_trees += 1
                if node == self.root:
                    print("This was the root's alt tree")

            for child in node.children.values():
                self.cleanup(child)
    return HAT()


def EFHAT():
    """Builds an Extremely Fast Hoeffding Adaptive Tree

    Returns:
        model: EFHAT
    """
    class Node:
        def __init__(self, is_leaf=True, prediction=None, is_alternate=False, parent=None, path=[]):
            self.is_leaf = is_leaf
            self.prediction = prediction
            self.split_feature = None
            self.split_value = None
            self.children = {}
            self.class_counts = np.zeros(2)
            self.adwin = ADWIN()
            self.alternate_tree = None
            self.n_predictions = 0
            self.n_correct_predictions = 0
            self.data_batch = []
            self.label_batch = []
            self.parent = parent
            self.path = tuple(path)
            self.is_alternate = is_alternate
            self.alt_order = 0
            self.counter = 0

        @property
        def accuracy(self):
            if self.n_predictions == 0:
                return 0
            return self.n_correct_predictions / self.n_predictions
        
    def hoeffding_bound(R, n):
        return np.sqrt((R**2 * np.log(1/0.10)) / (2 * n))

    def entropy(labels):
        # Count the occurrences of each class
        label_counts = np.bincount(labels, minlength=2)
        
        # Calculate the probabilities for each class
        probabilities = label_counts / np.sum(label_counts)
        
        # Remove probabilities equal to 0 for log2 calculation
        probabilities = probabilities[probabilities > 0]
        
        # Calculate the entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy 

    def information_gain(parent_labels, left_labels, right_labels):
        
        # Entropy before the split
        entropy_before = entropy(parent_labels)
        
        # Weighted entropy after the split
        total_size = len(parent_labels)
        left_size = len(left_labels)
        right_size = len(right_labels)
        
        weighted_entropy = (left_size / total_size) * entropy(left_labels) + \
                            (right_size / total_size) * entropy(right_labels)

        # Information gain is the reduction in entropy
        return entropy_before - weighted_entropy


    def best_split(data, labels):
        features = data.shape[1]
        best_info = {'feature': None, 'value': None, 'info_gain': -np.inf}
        second_best_info = {'feature': None, 'value': None, 'info_gain': -np.inf}
        
        # Iterate though all the features
        for feature in range(features):
            
            values = np.sort(np.unique(data[:, feature])) # Expand this
            values_n = len(values) -1
            
            # Sort through the unique values
            for i in range(values_n):
                split_value = (values[i] + values[i+1]) / 2
                
                # MArk the values with lower than the split
                smaller_values = data[:, feature] <= split_value
                #Negation fo the left
                bigger_values = ~smaller_values
                
                # Calculate Information Gain
                info_gain = information_gain(labels, labels[smaller_values], labels[bigger_values])
                
                if info_gain > best_info['info_gain']:
                    best_info, second_best_info = {
                        'feature': feature,
                        'value': split_value,
                        'info_gain': info_gain
                    }, best_info
                    
        return best_info, second_best_info
    class EFHAT:
        def __init__(self, batch_size=50):
            self.root = Node(is_leaf=True, prediction=0)
            self.alt_trees = 0
            self.pruned_trees = 0
            self.batch_size = batch_size
        
        def _fit_single(self, x, y):
            node = self.root
            while not node.is_leaf:
                if x[node.split_feature] < node.split_value:
                    self.HATGrow(x, y, node)
                    node = node.children['left']  
                elif x[node.split_feature] > node.split_value:
                    self.HATGrow(x, y, node)
                    node = node.children['right']
            
            node.data_batch.append(x)
            node.label_batch.append(y)
            node.class_counts[y] += 1
            node.prediction = np.argmax(node.class_counts)

            # Splitting
            if len(node.data_batch) >= self.batch_size:  
                self._attempt_to_split(node)
        
        def _attempt_to_split(self, node):
            X_sub = np.array(node.data_batch)
            y_sub = np.array(node.label_batch)

            if (node.class_counts[0] > 0 and node.class_counts[1] > 0):
                best_info, second_best_info = best_split(X_sub, y_sub)

                if best_info['feature'] is not None:
                    n = np.sum(node.class_counts)
                    epsilon = hoeffding_bound(1, n)
                    
                    if best_info['info_gain'] - 0 > epsilon: # G(X) is the info gain from an attribute X picked beforehand
                        newPath = list(node.path) + [node]
                        node.is_leaf = False
                        node.split_feature = best_info['feature']
                        node.split_value = best_info['value']
                        node.children['left'] = Node(is_leaf=True, prediction=np.argmax(node.class_counts), parent=node,
                                                is_alternate=node.is_alternate, path=newPath)
                        node.children['right'] = Node(is_leaf=True, prediction=np.argmax(node.class_counts), parent=node, 
                                                is_alternate=node.is_alternate, path=newPath)
                        self.split_accuracy_recalc(node)

            node.data_batch = []
            node.label_batch = []

        def HATGrow(self, x, y, node=None):
            # If node is None, start from root
            if node == None:
                node = self.root
            # While node is not a leaf, traverse the tree and check for alternate trees. If there is an alternate tree, pass it the data stream and grow it as well
            while not node.is_leaf:
                if x[node.split_feature] <= node.split_value:
                    if node.alternate_tree:
                        self.HATGrow(x, y, node.alternate_tree)
                    if node.is_leaf:
                        break
                    node = node.children['left']
                else:
                    if node.alternate_tree:
                        self.HATGrow(x, y, node.alternate_tree)
                    if node.is_leaf:
                        break
                    node = node.children['right']

            # Update the data_batch and label_batch of the node for best_split calculations later
            node.data_batch.append(x)
            node.label_batch.append(y)

            # Update that class counts and prediction of the leaf node
            correct = False
            node.class_counts[y] += 1
            node.prediction = np.argmax(node.class_counts)
            if node.prediction == y:
                correct = True
            node.n_predictions = np.sum(node.class_counts)
            node.n_correct_predictions = np.max(node.class_counts)

            # Update the accuracy of each node from the root to the leaf it got classified at
            self.update_whole_tree_accuracy(node, correct, y)

            # Split the node if the data_batch size is reached
            if len(node.data_batch) >= self.batch_size:  
                self._attempt_to_split(node)
            
            if node.parent:
                # For each node in the nodes path, update the estimators (ADWIN)
                for nodeX in node.path:
                    nodeX.adwin.update(1 if y != nodeX.prediction else 0)
                    # If drift is detected, create an alternate tree if there is not already one for that node and the node isnt an alternate itself.
                    if nodeX.adwin.drift_detected:
                        # If there is no alternate tree and the node is not a part of the alternate tree itself, create a alternate tree and recursively grow it.
                        if nodeX.alternate_tree is None and not node.is_alternate:
                            self.alt_trees += 1
                            node.alt_order = self.alt_trees
                            nodeX.alternate_tree = Node(is_leaf=True, prediction=np.argmax(nodeX.class_counts), is_alternate=True, parent=nodeX.parent)
                            self.HATGrow(x, y, nodeX.alternate_tree)
                            continue
                    
                    self.possibly_replace(nodeX)
            else:
                assert node == self.root or node == self.root.alternate_tree, "ERROR"
                node.adwin.update(1 if y != node.prediction else 0)
                if node.adwin.drift_detected:
                    # If there is no alternate tree and the node is not a part of the alternate tree itself, create a alternate tree and recursively grow it.
                    if node.alternate_tree is None and not node.is_alternate:
                        self.alt_trees += 1
                        node.alt_order = self.alt_trees
                        node.alternate_tree = Node(is_leaf=True, prediction=np.argmax(node.class_counts), is_alternate=True, parent=node.parent)
                        self.HATGrow(x, y, node.alternate_tree)
                self.possibly_replace(node)

        def possibly_replace(self, node):
            if node.alternate_tree:
                main_tree_accuracy = node.accuracy
                alt_tree_accuracy = node.alternate_tree.accuracy
                if alt_tree_accuracy > main_tree_accuracy:
                    self.pruned_trees += 1
                    node.is_leaf = node.alternate_tree.is_leaf
                    node.split_feature = node.alternate_tree.split_feature
                    node.split_value = node.alternate_tree.split_value
                    node.children = node.alternate_tree.children
                    node.n_correct_predictions = node.alternate_tree.n_correct_predictions
                    node.n_predictions = node.alternate_tree.n_predictions
                    print(f"Alternate Tree Replacing Old Tree at depth {len(node.path)}")
                    node.alternate_tree = None

        def update_is_alternate_status(self, children):
            for child in children.values():
                child.is_alternate = False
                self.update_is_alternate_status(child.children)

        # This method updates the accuracy of each node in the path that a piece of data followed down.
        # If the piece of data is classified correctly, the accuracy increases.
        def update_whole_tree_accuracy(self, node, correct, y):
            for x in node.path:
                x.class_counts[y] += 1
                x.prediction = np.argmax(x.class_counts)
                x.n_predictions += 1
                if correct:
                    x.n_correct_predictions += 1

        def split_accuracy_recalc(self, node):
            predictions = node.n_predictions
            correct_prediction = node.n_correct_predictions
            for x in node.path:
                x.n_predictions -= predictions
                x.n_correct_predictions -= correct_prediction
            node.n_predictions = 0
            node.n_correct_predictions = 0
        
        def predict(self, x):
            node = self.root
            while not node.is_leaf:
                if x[node.split_feature] <= node.split_value:
                    node = node.children['left']
                else:
                    node = node.children['right']
                    
            pred = node.prediction
            return pred
        
        def toString(self, node=None, depth=0):
            if node is None:
                node = self.root
            result = "  " * depth + f"Node(Leaf: {node.is_leaf}, Prediction: {node.prediction}, Alternate: {node.is_alternate}, "
            result += f"Split: [{node.split_feature}, {node.split_value}], N_Predictions: {node.n_predictions}, "
            result += f"N_Correct: {node.n_correct_predictions}, Class Counts: {node.class_counts.tolist()})\n"
            
            for child_key, child in node.children.items():
                result += "  " * (depth + 1) + f"{child_key.capitalize()} Child:\n"
                result += self.toString(child, depth + 2)
            
            return result

        def save_tree_to_file(self, filename='EFHAT_structure.txt'):
            tree_structure = self.toString()
            with open(filename, 'w') as file:
                file.write(tree_structure)
            print(f"Tree structure saved to {filename}")

        def cleanup(self, node=None):
            if node is None:
                node = self.root
            
            if node.alternate_tree and node.alternate_tree.accuracy <= node.accuracy:
                print("Pruning alternate tree")
                self.pruned_trees += 1
                if node == self.root:
                    print("This was the root's alt tree")

            for child in node.children.values():
                self.cleanup(child)
    return EFHAT()

if __name__ == '__main__':
    model1 = EFDT()
    model2 = HAT()
    model3 = EFHAT()

    file_path = './Skin_Data/Skin_NonSkin 2.txt'
    #file_path = './Skin_Data/Skin_NoSkin 1000.txt'
    #file_path = './Skin_Data/Skin_NoSkin 10000.txt'
    file_path = './Other_Data/cleaned_MiniBooNE_data.txt'

    # Load the data
    data = np.loadtxt(file_path, delimiter='\t')

    # Split the data into features and target variable
    X = data[:, :-1]
    y = data[:, -1].astype(int) 
    if file_path == './Skin_Data/Skin_NonSkin 2.txt':
        y = y - 1

    #Initialize variables
    errors_count = 0
    error_rates1 = []

    for idx in range(len(y)):
        pred = model1.predict(X[idx])

        if pred != y[idx]:
            errors_count += 1
            
        # Calculate Error Rate
        if idx > 10000:
            error_rate = errors_count / (idx + 1)
            error_rates1.append([idx, error_rate])
        
        # Print Every 10000 Iterations
        if idx % 10000 == 0:
            print(f'Instance: {idx}')
        
        # Fit the data point into the tree
        model1._fit_single(X[idx], y[idx])
    
    print(f'Final EFDT Error Rate: {error_rate}')

    errors_count = 0
    error_rates2 = []

    for i in range(len(y)):
        pred = model2.predict(X[i])

        if pred != y[i]:
            errors_count += 1
            
        # Calculate Error Rate
        if i > 10000:
            error_rate = errors_count / (i + 1)
            error_rates2.append([i, error_rate])
        
        # Print Every 10000 Iterations
        if i % 10000 == 0:
            print(f'Instance: {i}')
        
        model2.HATGrow(X[i], y[i])

    model2.cleanup()
    print(model2.pruned_trees)
    print(model2.alt_trees)
    model2.save_tree_to_file()

    print(f'Final HAT Error Rate: {error_rate}')

    errors_count = 0
    error_rates3 = []

    for i in range(len(y)):
        pred = model3.predict(X[i])

        if pred != y[i]:
            errors_count += 1
            
        # Calculate Error Rate
        if i > 10000:
            error_rate = errors_count / (i + 1)
            error_rates3.append([i, error_rate])
        
        # Print Every 10000 Iterations
        if i % 10000 == 0:
            print(f'Instance: {i}')
        
        model3._fit_single(X[i], y[i])

    model3.cleanup()
    print(model3.pruned_trees)
    print(model3.alt_trees)
    model3.save_tree_to_file()

    print(f'Final EFHAT Error Rate: {error_rate}')

    idxs, errors = zip(*error_rates1)
    i1, errors1 = zip(*error_rates2)
    i2, errors2 = zip(*error_rates3)

    plt.figure(figsize=(10, 6))
    plt.plot(idxs, errors, marker='o', linestyle='-', color='b', label='EFDT') 
    plt.plot(i1, errors1, marker='o', linestyle='-', color='r', label= 'HAT') 
    plt.plot(i2, errors2, marker='o', linestyle='-', color='g', label='EFHAT') 
    plt.xlabel('Iterations (x10,000)')
    plt.ylabel('Error Rate')
    plt.title('Error Rates of Models')
    plt.legend()
    plt.savefig('FinalImg.png')
    plt.grid(True)
    plt.show()