"""
Implements random forest objects

@author: Artem Naida
"""

# import decision_tree as dt
# import functions as fc


class RandomForest:
    """
    A random forest
    """

    def __init__(self):
        # creates a random forest with a set (req: 'odd' - so there are no ties) number of trees
        # trees are constructed with a random subset of features at each step
        # a data point is classified based on a vote by all trees
        pass

    def __str__(self):
        # A random forest with the following decision trees:
        # str(dt1)
        # str(dt2)
        # ...
        pass

    def view_graphic(self):
        # Show a graphic with each decision tree illustrated
        pass

    def view_data_path(self):
        # Highlight the path that a data point takes in each tree
        # Should show the total vote from each tree
        pass

    def classify_point(self):
        # Quickly classify a new point based on the random forest 
        pass

