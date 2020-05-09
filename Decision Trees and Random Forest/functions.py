"""
Implements all the functions that the decision tree and random forest objects require

@author: Artem Naida
"""

import numpy as np
import pandas as pd
from random import sample
from math import floor


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
    Finds the best split for continuous data using maximum information gain.

    Returns the value at which to split and the resulting information gain.

    :param data: pandas data frame
    :param feature: str
    :return: float, float
    """
    # If you send trivial data, give a trivial result
    input_entropy = entropy(data)
    if input_entropy == 0:
        return None, 0, "numeric"

    # Otherwise, get to work! Create pairs of values and labels to work with
    maximum_information_gain = 0
    pairs = []
    for pair in data[[feature, "lbl"]].values:
        pairs.append(tuple(pair))

    # If you pass just 2 data points, give a trivial split
    if len(pairs) == 2:
        midpoint = (pairs[0][0] + pairs[1][0]) / 2
        return midpoint, input_entropy, "numeric"

    # TODO: this while loop is just to delete 'x' and 'y' after use, is this necessary?
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

    # Just consider items in the overlap region
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

    # See if any point in the overlap region gives a better split
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
    :param data: pandas data frame
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


def best_split(input_data, random_subset=False):
    """
    Finds the feature that best splits the input data. If random subset is True, only consider a random
    subset of features.

    Returns the best feature, its type, the associated information gain, and the split
    :param input_data: pandas data frame
    :param random_subset: boolean
    :return: str, str, numeric, float
    :return: str, str, str, float
    """
    # TODO: just return None immediately if this function is passed trivial data
    # TODO: is it worth checking? Does the runtime get any better?
    # TODO: Repeat this behavior for the individual splitter functions as well - maybe at the aggregate level
    if random_subset:
        features = input_data.columns.tolist()
        del features[-1]
        n = len(features)
        selection = sample(features, floor(np.log2(n + 1)))
        selection.append("lbl")
        input_data = input_data[selection]
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
