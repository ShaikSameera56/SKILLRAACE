task2.(ii):

import math

def get_user_input(message):
    """Gets user input for a specific message."""
    while True:
        data = input(message)
        if data:
            return data.strip()
        else:
            print("Invalid input. Please enter a value.")

# Get user input for features and target name
features = []
while True:
    feature = get_user_input("Enter a feature name (or 'done' to finish): ")
    if feature.lower() == "done":
        break
    features.append(feature)

target_name = get_user_input("Enter the target name: ")

# Get user input for data
data = []
while True:
    row = []
    for feature in features:
        value = get_user_input(f"Enter value for '{feature}': ")
        row.append(value)
    play_decision = get_user_input(f"Is this a decision for '{target_name}' (True/False)? ")
    row.append(play_decision.lower() == "true")
    data.append(row)
    more_data = get_user_input("Enter more data (Yes/No)? ")
    if more_data.lower() != "yes":
        break

# Function to calculate Entropy (measure of impurity)
def entropy(data):
    labels = set([row[-1] for row in data])
    if len(labels) <= 1:
        return 0
    entropy = 0
    for label in labels:
        prob = sum([1 for row in data if row[-1] == label]) / len(data)
        entropy -= prob * math.log2(prob)
    return entropy

# Function to calculate Information Gain (reduction in entropy)
def information_gain(data, feature_index, threshold):
    left, right = [], []
    for row in data:
        if row[feature_index] <= threshold:
            left.append(row)
        else:
            right.append(row)
    parent_entropy = entropy(data)
    left_entropy = entropy(left) if len(left) > 0 else 0
    right_entropy = entropy(right) if len(right) > 0 else 0
    return parent_entropy - ((len(left) / len(data)) * left_entropy + (len(right) / len(data)) * right_entropy)

# Function to find the best split for a feature
def best_split(data, feature_index):
    best_gain = 0
    best_threshold = None
    values = set([row[feature_index] for row in data])
    for value in values:
        gain = information_gain(data, feature_index, value)
        if gain > best_gain:
            best_gain = gain
            best_threshold = value
    return best_gain, best_threshold

# Function to build the decision tree recursively
def build_tree(data, features):
    if len(data) == 0 or len(set([row[-1] for row in data])) == 1:
        return data[0][-1]  # Return the most frequent class label
    best_feature, best_threshold = None, None
    max_gain = 0
    for i in range(len(features)):
        gain, threshold = best_split(data, i)
        if gain > max_gain:
            max_gain = gain
            best_feature = i
            best_threshold = threshold
    left, right = [], []
    for row in data:
        if row[best_feature] <= best_threshold:
            left.append(row)
        else:
            right.append(row)
    if not features:
        return data[0][-1]  # Handle the case where all features are used
    return {
        features[best_feature]: {
            "<= " + str(best_threshold): build_tree(left, features[:best_feature] + features[best_feature + 1 :]),
            "> " + str(best_threshold): build_tree(right, features[:best_feature] + features[best_feature + 1 :])
        }
    }

# Build the decision tree
tree = build_tree(data, features.copy())

# Print the decision tree (can be replaced with visualization)
def print_tree(tree, level=0):
    if isinstance(tree, str):
        print(" " * level + tree)
    else:
        for key, value in tree.items():
            print(" " * level + str(key))
            print_tree(value, level + 1)

print_tree
