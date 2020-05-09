"""
Implements decision tree objects and decision tree node objects

@author: Artem Naida
"""

import functions as fc


class DecisionTreeNode:
    """
    A decision tree node
    """

    def __init__(self, input_data, random_subset=False):
        """
        Initializes a new Decision Tree Node. If random_subset is True, only chooses features from a random subset,
        newly created at each node.

        :param input_data: pandas data frame
        :param random_subset: boolean
        """
        self.input_data = input_data
        self.n = max(input_data.count())
        self.left, self.right = None, None
        # If you pass trivial input data or fewer than 10 items, just create a leaf
        if fc.entropy(input_data) == 0 or max(input_data.count()) < 10:
            self.leaf = True
            self.decision = None, None, None
            self.information_gain = 0
        else:
            result = fc.best_split(input_data, random_subset)
            feature_type = result[0]
            best_feature = result[1]
            split = result[2]
            self.information_gain = result[3]
            # If no information gain was possible, the node should be a leaf
            if self.information_gain == 0:
                self.leaf = True
                self.decision = None, None, None
            else:
                # Otherwise, node is not a leaf and has a decision type
                self.leaf = False
                self.decision = feature_type, best_feature, split

    def __str__(self):
        """
        Prints a human-readable representation of a Decision Tree Node. Currently very verbose.

        :return: str
        """
        if self.leaf:
            return "Decision Tree Node leaf taking '" + str(self.n) + \
                   "' data points with majority: '" + str(self.majority()) + "'"

        feature_type = self.decision[0]
        feature = self.decision[1]
        split = self.decision[2]
        gain = self.information_gain
        return "DecisionTreeNode with '" + str(self.n) + "' data points and '" \
               + str(feature_type) + "' decision type on feature '" + \
               str(feature) + "' split on '" + str(split) + \
               "' and information gain: " + str(gain) + " with majority: '" + str(self.majority()) + "'"

    def pass_right(self):
        """
        Takes a pandas data frame, passes the appropriate portion downstream to the right node.

        :return: pandas data frame
        """
        feature_type = self.decision[0]
        feature = self.decision[1]
        split = self.decision[2]
        if feature_type == "numeric":
            return self.input_data[self.input_data[feature] > split]
        elif feature_type == "categorical":
            return self.input_data[self.input_data[feature] == split]
        elif not feature_type:
            pass
        else:
            raise TypeError("Something went horribly wrong")

    def pass_left(self):
        """
        Takes a dataframe, passes the appropriate portion downstream to the left node.

        :return: pandas data frame
        """
        feature_type = self.decision[0]
        feature = self.decision[1]
        split = self.decision[2]
        if feature_type == "numeric":
            return self.input_data[self.input_data[feature] <= split]
        elif feature_type == "categorical":
            return self.input_data[self.input_data[feature] != split]
        elif not feature_type:
            pass
        else:
            raise TypeError("Something went horribly wrong")

    def send_datapoint(self, datapoint):
        """
        Takes a new data point as a single row pandas data frame.
        Returns 1 if the point goes right, 0 if it goes left.

        :param datapoint: pandas data frame
        :return: int
        """
        feature_type = self.decision[0]
        feature = self.decision[1]
        split = self.decision[2]
        if feature_type == "numeric":
            if datapoint[feature].values[0] > split:
                return 1
            return 0
        elif feature_type == "categorical":
            if datapoint[feature].values[0] == split:
                return 1
            return 0
        elif not feature_type:
            pass
        else:
            raise TypeError("Something went horribly wrong")

    def majority(self):
        label_list = self.input_data["lbl"].tolist()
        labels = set(label_list)
        max_item = None
        max_count = 0
        for item in labels:
            count = label_list.count(item)
            if count > max_count:
                max_item = item
                max_count = count
        return max_item

    def view_graphic(self):
        # Should display a box with condition for split written in text
        pass


class DecisionTree:
    """
    A decision tree.
    """

    def __init__(self, input_data, max_levels=None, random_subset=False):
        """
        Creates a new DecisionTree as a nested list of DecisionTreeNode objects organized by level. If random_subset
        is True, only chooses features from a random subset at each node.

        :param input_data: pandas data frame
        :param max_levels: int
        :param random_subset: boolean
        """
        self.nodes = [[DecisionTreeNode(input_data, random_subset)]]
        current_level = 0

        # This function checks whether all the nodes on your current level are leafs
        # TODO: can this function be defined elsewhere?
        def all_leafs(node_list):
            if not node_list:
                return False
            number_nodes = len(node_list)
            leafs = 0
            for node_ in node_list:
                if node_.leaf:
                    leafs += 1
            return number_nodes == leafs

        # Keep building a tree until all nodes on current level are leafs
        while not all_leafs(self.nodes[current_level]):
            next_level_list = []
            for node in self.nodes[current_level]:
                if not node.leaf:
                    left_node = DecisionTreeNode(node.pass_left(), random_subset)
                    right_node = DecisionTreeNode(node.pass_right(), random_subset)
                    node.left = left_node
                    node.right = right_node
                    next_level_list.append(left_node)
                    next_level_list.append(right_node)
            self.nodes.append(next_level_list)
            current_level += 1
            if max_levels and current_level == max_levels:
                break

    def __str__(self):
        """
        Prints the list of nodes in the decision tree.

        :return: str
        """
        # TODO: do we need this and print_tree?
        return str(self.nodes)

    def print_tree(self):
        """
        Prints a visual representation of the decision tree.

        :return: str
        """
        current_level = 0
        for level in self.nodes:
            print("[LEVEL: " + str(current_level) + "]")
            for node in level:
                print(node)
            current_level += 1

    def view_graphic(self):
        # displays a graphic of all the nodes and connections in the decision tree
        pass

    def classify_point(self, datapoint):
        """
        Classifies a new data point, represented as a pandas data frame with a single row.

        :param datapoint: pandas data frame
        :return: str
        """
        current_node = self.nodes[0][0]
        while not current_node.leaf:
            if current_node.send_datapoint(datapoint):
                current_node = current_node.right
            else:
                current_node = current_node.left
        return current_node.majority()

    def view_data_path(self):
        # takes a data point as input
        # display a graphic with the path that the point takes highlighted
        pass
