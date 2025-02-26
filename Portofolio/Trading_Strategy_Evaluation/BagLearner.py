import numpy as np
from scipy.stats import mode
class BagLearner(object):
    """
    This is a Bootstrap Aggregataion Learner (BagLearner).
    Parameters
        learner (learner) - Points to any arbitrary learner class that will be used in the BagLearner.
        kwargs            - Keyword arguments that are passed on to the learner’s constructor and they can vary according to the learner
        bags (int)        - The number of learners used to train using Bootstrap Aggregation.
                            If boost is true, then implement boosting (not implemented).
        verbose (bool)    - If “verbose” is True, print out information for debugging.
                            If verbose = False, do not generate ANY output.
    """
    def __init__(self, learner=object, kwargs = {}, bags = 1, boost=False, verbose=False):
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        learners = []
        for i in range(self.bags):
            learners.append(learner(**kwargs))
        self.learners = learners

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        Parameters
            data_x (numpy.ndarray) – A set of feature values used to train the learner
            data_y (numpy.ndarray) – The value we are attempting to predict given the X data
        """

        sample_num = data_x.shape[0]
        if len(data_y.shape) == 1:
            data_y = np.reshape(data_y, (data_x.shape[0], -1))
        data = np.column_stack((data_x, data_y))

        for learner in self.learners:
            data_temp = data[np.random.choice(sample_num, size=sample_num, replace=True), :]
            temp_x = data_temp[:, 0:- 1]
            temp_y = data_temp[:, -1]
            learner.add_evidence(temp_x, temp_y)  # call the learner's add_evidence method

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        Parameters
            points (numpy.ndarray) – A numpy array with each row corresponding to a specific query.
        Returns
            The predicted result of the input data according to the trained model
        Return type
            numpy.ndarray
        """
        # print(len(self.learners), points.shape)
        pred_y = np.zeros((points.shape[0], len(self.learners)))
        for i, learner in enumerate(self.learners):
           pred_y[:, i] = learner.query(points)

        result = mode(pred_y, axis=1)
        return result
    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "ycheng456"

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")