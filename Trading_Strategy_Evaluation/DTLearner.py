import numpy as np

class DTLearner(object):
    """
    This is a Decision Tree Learner (DTLearner).

    :param leaf_size: Is the maximum number of samples to be aggregated at a leaf, defaults to 1
    :type sd: integer

    :param verbose: If “verbose” is True, print out information for debugging.
        If verbose = False, not generate ANY output.
    :type verbose: boolean
    """
    def __init__(self, leaf_size=1, verbose=False):
      self.leaf_size = leaf_size
      self.verbose = verbose
      self.DTree = None

    def author(self):
       """
       :return: The GT username of the student
       :rtype: str
       """
       return "ycheng456"

    def best_factor(self, data_x, data_y):
        if data_x.ndim == 1:
            x_col = 1
        else:
            x_col = data_x.shape[1]
        corr_arr = []
        for i in range(x_col):
            corr_arr.append(np.corrcoef(data_x[:, i], data_y)[0, 1])
        correlation = np.absolute(corr_arr)
        return np.argmax(correlation)

    # build and save the model
    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size:
            return np.array([["leaf", np.mean(data_y), "NA", "NA"]])
        elif len(np.unique(data_y)) == 1:
            return np.array([["leaf", data_y[0], "NA", "NA"]])
        elif len(np.unique(data_x[0, :])) == 1:
            return np.array([["leaf", data_y[0], "NA", "NA"]])

            # Recursive case
        best_feature = self.best_factor(data_x, data_y)
        SplitVal = np.median(data_x[:, best_feature])

        left_indices = data_x[:, best_feature] <= SplitVal
        right_indices = data_x[:, best_feature] > SplitVal

        # Edge case
        if len(data_y[data_x[:, best_feature] <= SplitVal]) == 0 or len(
                data_y[data_x[:, best_feature] > SplitVal]) == 0:
            return np.array([["leaf", np.mean(data_y), "NA", "NA"]])

        lefttree = self.build_tree(data_x[data_x[:, best_feature] <= SplitVal], data_y[data_x[:, best_feature] <= SplitVal])
        righttree = self.build_tree(data_x[data_x[:, best_feature] > SplitVal], data_y[data_x[:, best_feature] > SplitVal])

        root = np.array([best_feature, SplitVal, 1, lefttree.shape[0] + 1])
        return np.vstack((np.vstack((root, lefttree)), righttree))

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        trained_tree = self.build_tree(data_x, data_y)
        self.DTree = trained_tree
        if self.verbose == True:
           print("Training Completed")
        return self.DTree

    def query(self, points):
        pred_y = np.zeros((points.shape[0],), dtype=float)
        for i in range(points.shape[0]):
            j = 0
            while j < self.DTree.shape[0]:
                if self.DTree[j][0] != "leaf":
                    var = self.DTree[j][0]
                    var_index = int(float(var))
                    if points[i, var_index] <= float(self.DTree[j][1]):
                        # left side
                        move_index = int(float(self.DTree[j][2]))
                    else:
                        move_index = int(float(self.DTree[j][3]))
                else:
                    pred_y[i] = float(self.DTree[j][1])
                    break
                j += move_index

        return pred_y

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
