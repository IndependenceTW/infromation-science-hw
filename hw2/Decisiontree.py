import sklearn.metrics
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
import numpy as np
import pandas as pd


class Node:
    "Decision tree node"

    def __init__(self, entropy, num_samples, num_samples_per_class, predicted_class, num_errors, alpha=float("inf")):
        self.entropy = entropy  # the entropy of current node
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class  # the majority class of the split group
        self.feature_index = 0  # the feature index we used to split the node
        self.threshold = 0  # for binary split
        self.left = None  # left child node
        self.right = None  # right child node
        self.num_errors = num_errors  # error after cut
        self.alpha = alpha  # each node alpha


class DecisionTreeClassifier:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth

    def _entropy(self, sample_y, n_classes):
        # TODO: calculate the entropy of sample_y and return it
        # sample_y represent the label of node
        # entropy = -sum(pi * log2(pi))
        _, counts = np.unique(sample_y, return_counts=True)

        total = sum(counts)
        probability = counts / total

        entropy = 0
        for i in range(probability.size):
            entropy += (probability[i] * math.log2(probability[i]))
        entropy = entropy * -1

        return entropy

    def _feature_split(self, X, y, n_classes):
        # Returns:
        #  best_idx: Index of the feature for best split, or None if no split is found.
        #  best_thr: Threshold to use for the split, or None if no split is found.
        m = y.size
        if m <= 1:
            return None, None

        # Entropy of current node.

        best_criterion = self._entropy(y, n_classes)

        best_idx, best_thr = None, None
        # TODO: find the best split, loop through all the features, and consider all the
        # midpoints between adjacent training samples as possible thresholds.
        # Compute the Entropy impurity of the split generated by that particular feature/threshold
        # pair, and return the pair with smallest impurity.

        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i)
                                 for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        correct_label_num = num_samples_per_class[predicted_class]
        num_errors = y.size - correct_label_num
        node = Node(
            entropy=self._entropy(y, self.n_classes_),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            num_errors=num_errors
        )

        if depth < self.max_depth:
            idx, thr = self._feature_split(X, y, self.n_classes_)
            if idx is not None:
                # TODO: Split the tree recursively according index and threshold until maximum depth is reached.
                pass
        return node

    def fit(self, X, Y):
        # TODO
        # Fits to the given training data
        pass

    def predict(self, X):
        pred = []
        # TODO: predict the label of data
        return pred

    def _find_leaves(self, root):
        # TODO
        # find each node child leaves number
        pass

    def _error_before_cut(self, root):
        # TODO
        # return error before post-pruning
        pass

    def _compute_alpha(self, root):
        # TODO
        # Compute each node alpha
        # alpha = (error after cut - error before cut) / (leaves been cut - 1)
        pass

    def _find_min_alpha(self, root):
        MinAlpha = float("inf")
        # TODO
        # Search the Decision tree which have minimum alpha's node
        pass

    def _prune(self):
        # TODO
        # prune the decision tree with minimum alpha node
        pass


def load_train_test_data(test_ratio=.3, random_state=1):
    df = pd.read_csv('./car.data', names=['buying', 'maint',
                     'doors', 'persons', 'lug_boot', 'safety', 'target'])
    X = df.drop(columns=['target'])
    X = np.array(X.values)
    y = np.array(df['target'].values)
    label = np.unique(y)
    # label encoding
    for i in range(len(y)):
        for j in range(len(label)):
            if y[i] == label[j]:
                y[i] = j
                break
    y = y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test


def accuracy_report(X_train_scale, y_train, X_test_scale, y_test, max_depth=7):
    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_train_scale, y_train)
    pred = tree.predict(X_train_scale)

    print(" tree train accuracy: %f"
          % (sklearn.metrics.accuracy_score(y_train, pred)))
    pred = tree.predict(X_test_scale)
    print(" tree test accuracy: %f"
          % (sklearn.metrics.accuracy_score(y_test, pred)))

    for i in range(10):
        print("=============Cut=============")
        tree._prune()
        pred = tree.predict(X_train_scale)
        print(" tree train accuracy: %f"
              % (sklearn.metrics.accuracy_score(y_train, pred)))
        pred = tree.predict(X_test_scale)
        print(" tree test accuracy: %f"
              % (sklearn.metrics.accuracy_score(y_test, pred)))


def main():
    X_train, X_test, y_train, y_test = load_train_test_data(
        test_ratio=.3, random_state=1)
    accuracy_report(X_train, y_train, X_test, y_test, max_depth=8)


if __name__ == "__main__":
    main()
