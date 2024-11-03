import pandas as pd
import numpy as np





def prune_tree(tree, data, target_attribute_name="class", min_samples=2):
    """
    Prune the decision tree by removing nodes with fewer than min_samples.
    """
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    subtree = tree[feature]

    for value in list(subtree.keys()):
        subset = data[data[feature] == value]
        if len(subset) < min_samples:
            # Replace subtree with the most common target value in the subset
            common_target = subset[target_attribute_name].mode()[0]
            subtree[value] = common_target
        else:
            subtree[value] = prune_tree(subtree[value], subset, target_attribute_name, min_samples)

    return tree


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy_value

def info_gain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    return information_gain

def id3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

# Training Examples:
training_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Target': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})
# training_data = pd.DataFrame({
#     'Feature1': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
#     'Feature2': ['X', 'X', 'Y', 'X', 'Y', 'Y', 'X', 'X', 'Y', 'Y'],
#     'Target': ['Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes']
# })
feature_columns = training_data.columns[:-1].to_list()
target_column = 'Target'
decision_tree = id3(training_data, training_data, feature_columns, target_column)
print("Original Decision Tree:",decision_tree)


pruned_tree = prune_tree(decision_tree, training_data, target_column, min_samples=6)
print("Pruned Tree:", pruned_tree)
