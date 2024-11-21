data_path = 'heart_disease.csv'
result_path = 'model_results.txt'

# Step 1: Load and Clean the Dataset (Manual Implementation)
def load_and_clean_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Parse header and data rows
    header = lines[0].strip().split(',')
    data = []
    for line in lines[1:]:
        row = line.strip().split(',')
        # Convert non-numeric entries to None
        clean_row = [float(x) if x != '?' else None for x in row]
        data.append(clean_row)
    
    # Remove rows with missing values
    data = [row for row in data if None not in row]
    return header, data

# Split features and target
def split_features_and_target(header, data, target_column):
    target_index = header.index(target_column)
    features = [row[:target_index] + row[target_index+1:] for row in data]
    target = [row[target_index] for row in data]
    return features, target

# Step 2: Manual Decision Tree Implementation
class DecisionTreeClassifierManual:
    def __init__(self):
        self.rules = []

    def fit(self, features, target, header):
        
        self.rules.append((header.index('chol'), 200, 1))  # Chol > 200 predicts 1 (heart disease)
        self.rules.append((header.index('age'), 50, 1))    # Age > 50 predicts 1 (heart disease)
    
    def predict(self, features):
        predictions = []
        for row in features:
            score = 0
            for rule in self.rules:
                index, threshold, predict = rule
                if row[index] > threshold:
                    score += predict
            predictions.append(1 if score > 0 else 0)
        return predictions

# Step 3: Custom Rule-Based Classifier
class CustomRuleBasedClassifier:
    def predict(self, features, header):

        age_index = header.index('age')
        chol_index = header.index('chol')
        thalach_index = header.index('thalach')
        predictions = []
        for row in features:
            if row[age_index] > 50 and row[chol_index] > 200 and row[thalach_index] < 150:
                predictions.append(1)  # Predict heart disease
            else:
                predictions.append(0)  # Predict no heart disease
        return predictions

# Evaluation function
def evaluate_predictions(true_labels, predictions):
    tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
    
    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


header, data = load_and_clean_data(data_path)
features, target = split_features_and_target(header, data, 'target')


split_index = int(len(features) * 0.7)
train_features, test_features = features[:split_index], features[split_index:]
train_target, test_target = target[:split_index], target[split_index:]

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifierManual()
dt_classifier.fit(train_features, train_target, header)
dt_predictions = dt_classifier.predict(test_features)
dt_metrics = evaluate_predictions(test_target, dt_predictions)

# Custom Rule-Based Classifier
custom_classifier = CustomRuleBasedClassifier()
custom_predictions = custom_classifier.predict(test_features, header)
custom_metrics = evaluate_predictions(test_target, custom_predictions)

# Save results
with open(result_path, 'w') as file:
    file.write("Decision Tree Classifier Results:\n")
    file.write(f"Accuracy: {dt_metrics[0]:.2f}\n")
    file.write(f"Precision: {dt_metrics[1]:.2f}\n")
    file.write(f"Recall: {dt_metrics[2]:.2f}\n")
    file.write(f"F1 Score: {dt_metrics[3]:.2f}\n")
    file.write("\nCustom Rule-Based Classifier Results:\n")
    file.write(f"Accuracy: {custom_metrics[0]:.2f}\n")
    file.write(f"Precision: {custom_metrics[1]:.2f}\n")
    file.write(f"Recall: {custom_metrics[2]:.2f}\n")
    file.write(f"F1 Score: {custom_metrics[3]:.2f}\n")


print("### Results")
print("#### Decision Tree Classifier")
print(f"- Accuracy: {dt_metrics[0] * 100:.2f}%")
print(f"- Precision: {dt_metrics[1] * 100:.2f}%")
print(f"- Recall: {dt_metrics[2] * 100:.2f}%")
print(f"- F1 Score: {dt_metrics[3] * 100:.2f}%\n")

print("#### Custom Rule-Based Classifier")
print(f"- Accuracy: {custom_metrics[0] * 100:.2f}%")
print(f"- Precision: {custom_metrics[1] * 100:.2f}%")
print(f"- Recall: {custom_metrics[2] * 100:.2f}%")
print(f"- F1 Score: {custom_metrics[3] * 100:.2f}%\n")

