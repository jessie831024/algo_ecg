import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.value = value  # Class label if the node is a leaf
        self.left = left  # Left subtree
        self.right = right  # Right subtree

class C45DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth  # Maximum depth of the tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.tree_ = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)
        most_common_class = unique_classes[np.argmax(counts)]

        # If only one class in the data or max depth reached, create a leaf node
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth) or num_samples < self.min_samples_split:
            return Node(value=most_common_class)

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build the subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature_index in range(num_features):
            feature_values = np.unique(X[:, feature_index])
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2  # Midpoints between unique values

            for threshold in thresholds:
                mask = X[:, feature_index] <= threshold
                gini = self._calculate_gini(y, mask)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, y, mask):
        left_gini = self._gini_impurity(y[mask])
        right_gini = self._gini_impurity(y[~mask])

        left_size = np.sum(mask)
        right_size = np.sum(~mask)
        total_size = left_size + right_size

        weighted_gini = (left_size / total_size) * left_gini + (right_size / total_size) * right_gini

        return weighted_gini

    def _gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree_) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)
