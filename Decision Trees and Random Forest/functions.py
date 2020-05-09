"""
Includes all the functions that the decision tree and random forest objects require

@author: Artem Naida
"""

import numpy as np
import pandas as pd


def entropy(data):
    """
    Calculates the information entropy of a data set

    :param data: pandas data frame
    :return: float
    """
    data_labels = data["lbl"].tolist()
    n = len(data_labels)
    total_entropy = 0
    for label in set(data_labels):
        label_count = data_labels.count(label)
        label_probability = label_count / n
        label_entropy = -1 * label_probability * np.log2(label_probability)
        total_entropy += label_entropy
    return total_entropy


def numeric_splitter(data, feature):
    """
    Finds the best split for continuous data using minimum resulting entropy.

    Returns the value at which to split and the resulting total resulting entropy.

    :param data: pandas dataframe
    :param feature: str
    :return: float, float
    """
    input_entropy = entropy(data)
    if input_entropy == 0:
        return None, 0, "numeric"
    maximum_information_gain = 0
    pairs = []
    for pair in data[[feature, "lbl"]].values:
        pairs.append(tuple(pair))
    # If you pass just 2 data points, give a trivial split
    if len(pairs) == 2:
        midpoint = (pairs[0][0] + pairs[1][0]) / 2
        return midpoint, input_entropy, "numeric"
    while 1:
        x = []
        y = []
        for item in pairs:
            if item[1] == pairs[0][1]:
                x.append(item[0])
            else:
                y.append(item[0])
        overlap = max(min(y), min(x)), min(max(y), max(x))
        break

    overlap_items = [item for item in pairs if (overlap[0] <= item[0] <= overlap[1])]
    overlap_middle = (overlap[0] + overlap[1]) / 2 # If no split reduces entropy, just returns this trivial split

    if not overlap_items: # If no overlapping items, return midpoint of empty region and associated entropy
        trivial_split = overlap_middle
        n = len(data.index)
        m = len(data[data[feature] > trivial_split].index)

        dfr1 = data[data[feature] > trivial_split]
        dfr2 = data[data[feature] <= trivial_split]

        trivial_split_entropy = (m / n) * entropy(dfr1) + ((n - m) / n) * entropy(dfr2)
        trivial_information_gain = input_entropy - trivial_split_entropy
        return trivial_split, trivial_information_gain, "numeric"

    best_split = overlap_middle
    for item in overlap_items:
        split = item[0]
        n = len(data.index)
        m = len(data[data[feature] > split].index)

        dfr1 = data[data[feature] > split]
        dfr2 = data[data[feature] <= split]

        resulting_entropy = (m / n) * entropy(dfr1) + ((n - m) / n) * entropy(dfr2)
        current_information_gain = input_entropy - resulting_entropy

        if current_information_gain > maximum_information_gain:
            best_split = split
            maximum_information_gain = current_information_gain

    return best_split, maximum_information_gain, "numeric"


def categorical_splitter(data, feature):
    """
    Chooses the group within the categorical feature that results in a split with minimum entropy.
    The resulting paths split the data by those points which are in the output group and those which are not.

    Returns the chosen 'in' group and the total data entropy associated with the split.
    :param data: pandas dataframe
    :param feature: str
    :return: str, float
    """
    input_entropy = entropy(data)
    max_information_gain = 0
    current_type = ""
    groups = set(data[feature])
    for group in groups:
        n = len(data.index)
        m = len(data[data[feature] == group].index)

        dfr1 = data[data[feature] == group]
        dfr2 = data[data[feature] != group]

        resulting_entropy = (m / n) * entropy(dfr1) + ((n - m) / n) * entropy(dfr2)
        information_gain = input_entropy - resulting_entropy

        if information_gain > max_information_gain:
            current_type = group
            max_information_gain = information_gain

    return current_type, max_information_gain, "categorical"


def splitter(data, feature):
    """
    Finds the best split for the data based on feature. Sends data to numerical splitter or categorical splitter based
    on whether it is appropriate for either.

    Returns 'in' group from categorical split, or split point from numerical split, as well as resulting information
    gain and type of splitting variable.
    :param data: pandas data frame
    :param feature: str
    :return: str, float, str
    :return: float, float, str
    """
    if pd.api.types.is_string_dtype(data[feature]):
        return categorical_splitter(data, feature)
    elif pd.api.types.is_numeric_dtype(data[feature]):
        return numeric_splitter(data, feature)
    else:
        raise TypeError("'splitter' function requires a numeric or string type feature column as second arg")


def best_split(input_data):
    """
    Finds the feature that best splits the input data

    Returns the best feature, its type, the associated information gain, and the split
    :param input_data: pandas data frame
    :return:
    """
    # TODO: just return None immediately if this function is passed trivial data
    # Repeat this behavior for the individual splitter functions as well - maybe at the aggregate level
    best_feature = None
    feature_type = None
    maximum_information_gain = 0
    split = None
    for feature in input_data.columns:
        if feature == "lbl":
            break
        result = splitter(input_data, feature)
        if result[1] > maximum_information_gain:
            best_feature = feature
            feature_type = result[2]
            maximum_information_gain = result[1]
            split = result[0]
    return feature_type, best_feature, split, maximum_information_gain
