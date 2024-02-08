import numpy as np
from graphviz import Digraph

class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

class C45DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.tree_ = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)
        most_common_class = unique_classes[np.argmax(counts)]

        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth) or num_samples < self.min_samples_split:
            return Node(value=most_common_class)

        best_feature, best_threshold = self._find_best_split(X, y)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_info_gain = -float('inf')
        best_feature = None
        best_threshold = None

        for feature_index in range(num_features):
            feature_values = np.unique(X[:, feature_index])
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2

            for threshold in thresholds:
                mask = X[:, feature_index] <= threshold
                info_gain = self._calculate_info_gain(y, mask)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_info_gain(self, y, mask):
        parent_entropy = self._calculate_entropy(y)
        left_entropy = self._calculate_entropy(y[mask])
        right_entropy = self._calculate_entropy(y[~mask])

        left_size = np.sum(mask)
        right_size = np.sum(~mask)
        total_size = left_size + right_size

        weighted_entropy = (left_size / total_size) * left_entropy + (right_size / total_size) * right_entropy

        info_gain = parent_entropy - weighted_entropy
        return info_gain

    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree_) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)


    def export_tree_text(self):
        tree_rules = self._export_tree_text(self.tree_)
        return tree_rules

    def _export_tree_text(self, node, depth=0):
        indent = "  " * depth
        if node.value is not None:
            return f"{indent}Class: {node.value}\n"
        else:
            left_rules = self._export_tree_text(node.left, depth + 1)
            right_rules = self._export_tree_text(node.right, depth + 1)
            return f"{indent}if feature {node.feature_index} <= {node.threshold}:\n{left_rules}" \
                   f"{indent}else:\n{right_rules}"

