import sklearn.metrics
from sklearn.model_selection import train_test_split
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
    def __init__(self, max_depth=4, n_classes=2):
        self.max_depth = max_depth
        self.n_classes_ = n_classes
        self.root = None

    def _entropy(self, sample_y):
        # calculate the entropy of sample_y and return it
        # sample_y represent the label of node
        # entropy = -sum(pi * log2(pi))
        labels = np.unique(sample_y)
        entropy = 0
        for label in labels:
            p = len(sample_y[sample_y == label]) / len(sample_y)
            entropy += p * np.log2(p)
        entropy *= -1
        return entropy

    def _feature_split(self, X, y):
        # Returns:
        #  best_idx: Index of the feature for best split, or None if no split is found.
        #  best_thr: Threshold to use for the split, or None if no split is found.
        m = y.size
        if m <= 1:
            return None, None

        class_y = np.unique(y)
        if len(class_y) <= 1:
            return None, None

        # Entropy of current node.

        entropy_current = self._entropy(y)
        best_inf_gain = -float("inf")

        best_idx, best_thr = None, None
        # find the best split, loop through all the features, and consider all the
        # midpoints between adjacent training samples as possible thresholds.
        # Compute the Entropy impurity of the split generated by that particular feature/threshold
        # pair, and return the pair with smallest impurity.
        feature_num = np.shape(X)[1]
        for feature_idx in range(feature_num):
            feature_val = X[:, feature_idx]
            possible_thrs = np.unique(feature_val)
            for thr in possible_thrs:
                _, _, left_y, right_y = self._split(
                    X, y, feature_idx, thr)
                entropy_after_split = (len(left_y) / len(y)) * self._entropy(
                    left_y) + (len(right_y) / len(y)) * self._entropy(right_y)
                inf_gain = entropy_current - entropy_after_split
                if (inf_gain > best_inf_gain):
                    best_inf_gain = inf_gain
                    best_idx = feature_idx
                    best_thr = thr
        return best_idx, best_thr

    def _split(self, target_set, target_y, feature_idx, thr):
        left_idx = [i for i in range(
            len(target_y)) if target_set[i][feature_idx] is thr]
        left = np.array(target_set[left_idx])
        left_y = np.array(target_y[left_idx])
        right_idx = [i for i in range(
            len(target_y)) if target_set[i][feature_idx] is not thr]
        right = np.array(target_set[right_idx])
        right_y = np.array(target_y[right_idx])
        return left, right, left_y, right_y

    def _build_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i)
                                 for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        correct_label_num = num_samples_per_class[predicted_class]
        num_errors = y.size - correct_label_num
        node = Node(
            entropy=self._entropy(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            num_errors=num_errors
        )

        if depth < self.max_depth:
            idx, thr = self._feature_split(X, y)
            if idx is not None:
                # Split the tree recursively according index and threshold until maximum depth is reached.
                left_X, right_X, left_y, right_y = self._split(X, y, idx, thr)
                node.threshold = thr
                node.feature_index = idx
                node.left = self._build_tree(left_X, left_y, depth+1)
                node.right = self._build_tree(right_X, right_y, depth+1)
        return node

    def fit(self, X, Y):
        # Fits to the given training data
        self.root = self._build_tree(X, Y)

    def predict(self, X):
        pred = []
        # predict the label of data
        for x in X:
            y = self._tree_traverse(x, self.root)
            pred.append(y)
        return pred

    def _tree_traverse(self, x, root):
        if (root.left is None):
            return root.predicted_class
        feature_val = x[root.feature_index]
        if (feature_val == root.threshold):
            return self._tree_traverse(x, root.left)
        else:
            return self._tree_traverse(x, root.right)

    def _find_leaves(self, root):
        # find each node child leaves number
        if (root.left is None):
            return 1
        else:
            return self._find_leaves(root.left) + self._find_leaves(root.right)

    def _error_before_cut(self, root):
        # return error before post-pruning
        if (root.left is None):
            return root.num_errors
        else:
            return self._error_before_cut(root.left) + self._error_before_cut(root.right)

    def _compute_alpha(self, root):
        # Compute each node alpha
        # alpha = (error after cut - error before cut) / (leaves been cut - 1)
        if (root.left is not None):
            root.alpha = (
                root.num_errors - self._error_before_cut(root)) / (self._find_leaves(root) - 1)
            self._compute_alpha(root.left)
            self._compute_alpha(root.right)
        else:
            root.alpha = float("inf")

    def _find_min_alpha(self, root):
        MinAlpha = float("inf")
        # Search the Decision tree which have minimum alpha's node
        queue = []
        queue.append(root)
        ret = None

        while (len(queue) != 0):
            curr_node = queue.pop(0)
            if (curr_node.left is not None):
                queue.append(curr_node.left)
                queue.append(curr_node.right)
            if (curr_node.alpha <= MinAlpha):
                MinAlpha = curr_node.alpha
                ret = curr_node

        return ret

    def _prune(self):
        # prune the decision tree with minimum alpha node
        self._compute_alpha(self.root)
        cut_node = self._find_min_alpha(self.root)
        cut_node.left = None
        cut_node.right = None


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
    tree = DecisionTreeClassifier(
        max_depth=max_depth, n_classes=len(np.unique(y_train)))
    tree.fit(X_train_scale, y_train)
    pred = tree.predict(X_train_scale)

    # print("tree train leaves: " + str(tree._find_leaves(tree.root)))

    print(" tree train accuracy: %f"
          % (sklearn.metrics.accuracy_score(y_train, pred)))
    pred = tree.predict(X_test_scale)
    print(" tree test accuracy: %f"
          % (sklearn.metrics.accuracy_score(y_test, pred)))
    for i in range(10):
        print("=============Cut=============")
        tree._prune()
        pred = tree.predict(X_train_scale)
        # print("tree train leaves: " + str(tree._find_leaves(tree.root)))
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
