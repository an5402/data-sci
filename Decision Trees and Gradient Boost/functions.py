"""
Implements all the functions required for decision_tree and gradient_boost

@author: Artem Naida
"""

import pandas as pd
import numpy as np


def split_continuous(data, feat):
    """
    Splits continuous data based on maximum variance reduction.
    Returns the best splitting point, along with the maximum variance reduction and the split type identifier.

    :param data: pandas data frame
    :param feat: str
    :return: numeric, float, str
    """
    points = data[feat]
    input_variance = np.var(data["tg"])
    maximum_variance_reduction = 0
    best_split = None
    for point in points:
        dfr1 = data[data[feat] > point]
        dfr2 = data[data[feat] <= point]

        n = len(dfr1.index)
        m = len(dfr2.index)

        resulting_variance = (n / (n + m)) * np.var(dfr1["tg"]) + (m / (n + m)) * np.var(dfr2["tg"])
        variance_reduction = input_variance - resulting_variance

        if variance_reduction > maximum_variance_reduction:
            best_split = point
            maximum_variance_reduction = variance_reduction

    return best_split, maximum_variance_reduction, "continuous"


def split_categorical(data, feat):
    """
    Splits categorical data based on maximum variance reduction.
    Returns the 'in' level that gives the best split, along with the maximum variance reduction and the
    split type identifier.

    :param data: pandas data frame
    :param feat: str
    :return: str, float, str
    """
    maximum_variance_reduction = 0
    input_variance = np.var(data["tg"])
    best_split = None

    levels = set(data[feat].tolist())
    for level in levels:
        dfr1 = data[data[feat] == level]
        dfr2 = data[data[feat] != level]

        n = len(dfr1.index)
        m = len(dfr2.index)

        resulting_variance = (n / (n + m)) * np.var(dfr1["tg"]) + (m / (n + m)) * np.var(dfr2["tg"])
        variance_reduction = input_variance - resulting_variance

        if variance_reduction > maximum_variance_reduction:
            best_split = level
            maximum_variance_reduction = variance_reduction

    return best_split, maximum_variance_reduction, "categorical"


# noinspection PyUnresolvedReferences
def splitter(data, feature):
    """
    Identifies the type of split to perform and sends to the appropriate splitter function.
    Returns the result of that splitter function.

    :param data: pandas data frame
    :param feature: str
    :return: str, float, str
    :return: numeric, float, str
    """
    if feature not in data.columns:
        raise ValueError("Feature must be a valid column name from data")
    if pd.api.types.is_string_dtype(data[feature]):
        return split_categorical(data, feature)
    elif pd.api.types.is_numeric_dtype(data[feature]):
        return split_continuous(data, feature)
    else:
        raise TypeError("Feature must be numeric or categorical")  # This shouldn't ever run


def find_best_split(data):
    """
    Finds the best feature to split the data on by way of variance reduction.
    Returns the best feature and the associated variance reduction.

    :param data: pandas data frame
    :return: str, float
    """
    max_variance_reduction = 0
    best_feature, type_, split = None, None, None
    for feature in data.columns:
        if feature == "tg":
            break
        result = splitter(data, feature)
        variance_reduction = result[1]
        if variance_reduction > max_variance_reduction:
            best_feature = feature
            split = result[0]
            type_ = result[2]
            max_variance_reduction = variance_reduction
    return type_, best_feature, split, max_variance_reduction
