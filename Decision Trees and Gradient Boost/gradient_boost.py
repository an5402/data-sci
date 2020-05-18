"""
Implements the functionality for gradient boost, a method for estimating a regression relationship.

@author: Artem Naida
"""

import decision_tree as dt


class GradientBoost:
    """
    A Gradient Boost algorithm. Uses a series of scaled and leaf-limited decision trees
    to refine the residuals from a regression.
    """
    def __init__(self, input_data, learning_rate, max_number_trees, max_number_leaves):
        self.data = input_data
        initial_prediction = dt.fc.np.mean(input_data["tg"])
        self.data["tg"] = self.data["tg"] - initial_prediction
        self.trees = [dt.DecisionTreeReg(self.data, max_leaves=max_number_leaves)]
        # From here on things get weird...
        for i in range(2, max_number_trees + 2):
            self.data["tg"] = self.data["tg"] - learning_rate * self.trees[i - 2].predict_point(self.data)
            self.trees.append(dt.DecisionTreeReg(self.data, max_leaves=max_number_leaves))

    def predict_point(self, point):
        # Predict the value of a new point using the gradient boost algorithm
        pass
