"""
Implements random forest objects

@author: Artem Naida
"""

import decision_tree as dt


class RandomForest:
    """
    A random forest
    """

    def __init__(self, input_data, number_trees):
        """
        Creates a new random forest as a list of decision trees.
        Number of trees must be an odd positive integer.

        :param input_data: pandas data frame
        :param number_trees: int
        """
        if number_trees % 2 == 0 or number_trees <= 0:
            raise ValueError("Number of trees must be an odd positive integer")
        self.trees = []
        for i in range(0, number_trees):
            bootstrapped_data = input_data.sample(max(input_data.count()), replace=True)
            self.trees.append(dt.DecisionTree(bootstrapped_data, random_subset=True))

    def __str__(self):
        """
        Prints a simple representation of the random forest.

        :return: str
        """
        return "A random forest with the following decision trees: " + str(self.trees)

    def print_forest(self):
        """
        Prints a detailed description of the random forest and each tree.

        :return: str
        """
        i = 1
        for tree in self.trees:
            print("[TREE: " + str(i) + "]")
            tree.print_tree()
            i += 1

    def view_graphic(self):
        # Show a graphic with each decision tree illustrated
        pass

    def view_data_path(self):
        # Highlight the path that a data point takes in each tree
        # Should show the total vote from each tree
        pass

    def classify_point(self, datapoint):
        """
        Classifies a data point with the random forest. Decision based on a vote from each tree.
        Data point should be a pandas data frame with a single row.

        :param datapoint:
        :return: str
        """
        votes = []
        for tree in self.trees:
            votes.append(tree.classify_point(datapoint))
        max_count = 0
        max_item = None
        items = set(votes)
        for item in items:
            count = votes.count(item)
            if count > max_count:
                max_item = item
                max_count = count
        return max_item
