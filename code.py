import math
import random

# Load and preprocess the dataset
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            features = list(map(float, parts[2:]))
            label = 1 if parts[1] == 'M' else 0
            data.append((features, label))
    return data

# Split dataset into features (X) and labels (y)
def split_features_labels(data):
    X = [row[0] for row in data]
    y = [row[1] for row in data]
    return X, y

# Split data into training and testing sets
def train_test_split(data, test_ratio=0.2):
    random_indices = list(range(len(data)))
    random.shuffle(random_indices)
    split_point = int(len(data) * (1 - test_ratio))
    train_data = [data[i] for i in random_indices[:split_point]]
    test_data = [data[i] for i in random_indices[split_point:]]
    return train_data, test_data

# Accuracy calculation
def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Precision, Recall, and F1-Score
def calculate_metrics(y_true, y_pred):
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score

# SVM Implementation
class SVM:
    def __init__(self, learning_rate=0.001, epochs=1000, regularization=0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.reg = regularization
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.w = [0] * n_features
        self.b = 0
        y_ = [1 if yi == 1 else -1 for yi in y]
        losses = []

        for epoch in range(self.epochs):
            loss = 0
            for idx in range(n_samples):
                x_i = X[idx]
                condition = y_[idx] * (self.dot_product(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w = [w - self.lr * 2 * self.reg * w for w in self.w]
                else:
                    self.w = [
                        w - self.lr * (2 * self.reg * w - x_i[i] * y_[idx])
                        for i, w in enumerate(self.w)
                    ]
                    self.b -= self.lr * y_[idx]
                    loss += max(0, 1 - y_[idx] * (self.dot_product(x_i, self.w) + self.b))
            losses.append(loss)

    def predict(self, X):
        return [1 if self.dot_product(x, self.w) + self.b >= 0 else 0 for x in X]

    def dot_product(self, x1, x2):
        return sum(a * b for a, b in zip(x1, x2))

# Random Forest Implementation
class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            return max(set(y), key=y.count)

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return max(set(y), key=y.count)

        left_idx = [i for i in range(len(X)) if X[i][feature_idx] < threshold]
        right_idx = [i for i in range(len(X)) if i not in left_idx]

        left = self._grow_tree([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1)
        right = self._grow_tree([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1)
        return (feature_idx, threshold, left, right)

    def _best_split(self, X, y):
        best_gain = 0
        split_idx, split_thresh = None, None

        for i in range(len(X[0])):
            thresholds = set(row[i] for row in X)
            for t in thresholds:
                gain = self._information_gain(y, X, i, t)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, i, t
        return split_idx, split_thresh

    def _information_gain(self, y, X, feature_idx, threshold):
        parent_entropy = self._entropy(y)
        left = [y[i] for i in range(len(y)) if X[i][feature_idx] < threshold]
        right = [y[i] for i in range(len(y)) if X[i][feature_idx] >= threshold]

        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(y)
        child_entropy = (len(left) / n) * self._entropy(left) + (len(right) / n) * self._entropy(right)
        return parent_entropy - child_entropy

    def _entropy(self, y):
        counts = {label: y.count(label) for label in set(y)}
        total = len(y)
        return -sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0)

    def predict(self, X):
        return [self._traverse_tree(x, self.tree) for x in X]

    def _traverse_tree(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature_idx, threshold, left, right = tree
        if x[feature_idx] < threshold:
            return self._traverse_tree(x, left)
        return self._traverse_tree(x, right)

class RandomForest:
    def __init__(self, n_trees=5, max_depth=5):
        self.n_trees = n_trees
        self.trees = [DecisionTree(max_depth) for _ in range(n_trees)]

    def fit(self, X, y):
        for tree in self.trees:
            sampled_indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
            tree.fit([X[i] for i in sampled_indices], [y[i] for i in sampled_indices])

    def predict(self, X):
        tree_preds = [tree.predict(X) for tree in self.trees]
        final_preds = []
        for i in range(len(X)):
            votes = [pred[i] for pred in tree_preds]
            final_preds.append(max(set(votes), key=votes.count))
        return final_preds

# Main Function
data = load_data('wdbc.data')
X, y = split_features_labels(data)
train_data, test_data = train_test_split(data)
X_train, y_train = split_features_labels(train_data)
X_test, y_test = split_features_labels(test_data)

# Train and evaluate SVM
svm = SVM()
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
svm_accuracy = calculate_accuracy(y_test, svm_predictions)
svm_precision, svm_recall, svm_f1 = calculate_metrics(y_test, svm_predictions)

# Train and evaluate Random Forest
rf = RandomForest(n_trees=5, max_depth=5)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
rf_accuracy = calculate_accuracy(y_test, rf_predictions)
rf_precision, rf_recall, rf_f1 = calculate_metrics(y_test, rf_predictions)

# Print Results
print(f"SVM: Accuracy={svm_accuracy:.2f}, Precision={svm_precision:.2f}, Recall={svm_recall:.2f}, F1-Score={svm_f1:.2f}")
print(f"Random Forest: Accuracy={rf_accuracy:.2f}, Precision={rf_precision:.2f}, Recall={rf_recall:.2f}, F1-Score={rf_f1:.2f}")
 # type: ignore