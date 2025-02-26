import numpy as np
from scipy.stats import mode
from scipy.stats.stats import ModeResult

class RTLearner(object):
    """
    This is a Random Tree Learner (DTLearner).

    :param leaf_size: Is the maximum number of samples to be aggregated at a leaf, defaults to 1
    :type sd: integer

    :param verbose: If “verbose” is True, print out information for debugging.
        If verbose = False, not generate ANY output.
    :type verbose: boolean
    """


    def __init__(self, leaf_size=5, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.RTree = None

    def author(self):
       """
       :return: The GT username of the student
       :rtype: str
       """
       return "ycheng456"

    # find a random feature as split variable
    def rand_factor(self, data_x, data_y):
        x_num = data_x.shape[1]
        feat_index = np.random.randint(x_num)
        return feat_index

        # build and save the model
    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size:
            return np.array([["leaf", mode(data_y), "NA", "NA"]])
        elif len(np.unique(data_y)) == 1:
            return np.array([["leaf", data_y[0], "NA", "NA"]])
        elif len(np.unique(data_x[0, :])) == 1:
            return np.array([["leaf", data_y[0], "NA", "NA"]])

            # Recursive case
        rand_feature = self.rand_factor(data_x, data_y)
        SplitVal = np.median(data_x[:, rand_feature])

        left_indices = data_x[:, rand_feature] <= SplitVal
        right_indices = data_x[:, rand_feature] > SplitVal

        # Edge case
        if len(data_y[data_x[:, rand_feature] <= SplitVal]) == 0 or len(
                data_y[data_x[:, rand_feature] > SplitVal]) == 0:
            return np.array([["leaf", mode(data_y), "NA", "NA"]])

        lefttree = self.build_tree(data_x[data_x[:, rand_feature] <= SplitVal], data_y[data_x[:, rand_feature] <= SplitVal])
        righttree = self.build_tree(data_x[data_x[:, rand_feature] > SplitVal], data_y[data_x[:, rand_feature] > SplitVal])

        root = np.array([rand_feature, SplitVal, 1, lefttree.shape[0] + 1])
        return np.vstack((np.vstack((root, lefttree)), righttree))

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        trained_tree = self. build_tree(data_x, data_y)
        self.RTree = trained_tree
        if self.verbose == True:
           print("Training Completed")
        return self.RTree

    def query(self, points):
        # Expecting x values 1-3 for example
        # Tree is a series of 1x4 arrays: [Factor, Split Value, Left node, Right Node
        pred_y = np.zeros((points.shape[0],), dtype=float)
        for i in range(points.shape[0]):
            j = 0
            while j < self.RTree.shape[0]:
                # pull relevant factor from points
                var = self.RTree[j][0]
                if var != "leaf":
                    var_index = int(float(var))
                    if points[i, var_index] <= float(self.RTree[j][1]):
                        # left side
                        move_index = int(float(self.RTree[j][2]))
                    else:
                        move_index = int(float(self.RTree[j][3]))
                else:
                    # once the leaf is found return y value
                    if type(self.RTree[j][1]) == ModeResult:
                        pred_y[i] = self.RTree[j][1].mode[0]
                    else:
                        pred_y[i] = self.RTree[j][1]
                    break
                j += move_index

        return pred_y

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")