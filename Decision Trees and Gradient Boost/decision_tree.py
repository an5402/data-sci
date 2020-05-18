"""
Implements decision tree regression objects

@author: Artem Naida
"""

import functions as fc


class DecisionTreeRegNode:
    """
    A node in a decision tree for regression
    """
    def __init__(self, data):
        """
        A node in a Decision Tree for Regression.
        """
        self.input_data = data
        self.n = max(data.count())
        self.left, self.right = None, None

        # If you pass trivial input data or fewer than 10 items, just create a leaf
        if self.n < 10 or fc.np.var(data["tg"]) == 0:
            self.leaf = True
            self.decision = None, None, None
            self.variance_reduction = 0
        else:
            result = fc.find_best_split(data)
            if result[3] == 0:
                self.leaf = True
                self.decision = None, None, None
                self.variance_reduction = 0

            best_feature_type = result[0]
            best_feature = result[1]
            split = result[2]
            self.variance_reduction = result[3]
            self.leaf = False
            self.decision = best_feature_type, best_feature, split

    def __str__(self):
        """
        Prints a human-readable representation of a node in a decision tree for regression.

        :return: str
        """
        if self.leaf:
            return "A decision tree for regression leaf node with prediction: " + str(self.predict())
        type_ = self.decision[0]
        feature = self.decision[1]
        split = self.decision[2]
        variance_reduction = self.variance_reduction
        return "A decision tree for regression node split on " + str(type_) + " feature type named " + \
               str(feature) + " split on " + str(split) + " with variance reduction " + str(variance_reduction)

    def pass_right(self):
        """
        Takes a pandas data frame, passes the appropriate portion downstream to the right node.

        :return: pandas data frame
        """
        feature_type = self.decision[0]
        feature = self.decision[1]
        split = self.decision[2]
        if feature_type == "continuous":
            return self.input_data[self.input_data[feature] > split]
        elif feature_type == "categorical":
            return self.input_data[self.input_data[feature] == split]
        elif not feature_type:
            pass
        else:
            raise TypeError("Something went horribly wrong")

    def pass_left(self):
        """
        Takes a pandas data frame, passes the appropriate portion downstream to the left node.

        :return: pandas data frame
        """
        feature_type = self.decision[0]
        feature = self.decision[1]
        split = self.decision[2]
        if feature_type == "continuous":
            return self.input_data[self.input_data[feature] <= split]
        elif feature_type == "categorical":
            return self.input_data[self.input_data[feature] != split]
        elif not feature_type:
            pass
        else:
            raise TypeError("Something went horribly wrong")

    def predict(self):
        """
        Produces the prediction of the target variable based on the current data.
        Returns the prediction as the mean of the target variable within the input data.

        :return: float
        """
        return fc.np.mean(self.input_data["tg"])

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
        if feature_type == "continuous":
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


class DecisionTreeReg:
    """
    A decision tree for regression
    """
    def __init__(self, input_data, max_leaves=None):
        """
        Creates a new DecisionTreeReg as a nested list of DecisionTreeRegNode objects organized by level. If
        random_subset is True, only chooses features from a random subset at each node.

        :param input_data: pandas data frame
        :param max_leaves: int
        """
        self.nodes = [[DecisionTreeRegNode(input_data)]]
        current_leaves = 1
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
                if max_leaves and current_leaves == max_leaves:
                    break
                # If the node is not a leaf, split it
                if not node.leaf:
                    left_node = DecisionTreeRegNode(node.pass_left())
                    right_node = DecisionTreeRegNode(node.pass_right())
                    node.left = left_node
                    node.right = right_node
                    next_level_list.append(left_node)
                    next_level_list.append(right_node)
                    current_leaves += 1
            self.nodes.append(next_level_list)
            current_level += 1
            # If we have reached max leaves, break
            if max_leaves and current_leaves == max_leaves:
                break

        for level in self.nodes:
            for node in level:
                if not node.left and not node.right:
                    node.leaf = True

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

    def predict_point(self, datapoint):
        """
        Predicts the target value for a new data point, represented as a pandas data frame with a single row.

        :param datapoint: pandas data frame
        :return: float
        """
        current_node = self.nodes[0][0]
        while not current_node.leaf:
            if current_node.send_datapoint(datapoint):
                current_node = current_node.right
            else:
                current_node = current_node.left
        return current_node.predict()

    def view_graphic(self):
        # displays a graphic of all the nodes and connections in the decision tree
        pass

    def view_data_path(self):
        # takes a data point as input
        # display a graphic with the path that the point takes highlighted
        pass

