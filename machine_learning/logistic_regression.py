import numpy as np


class LogisticRegression:
    """Logistic Regression classifier.

    Logistic Regression classification using the sigmoid function to predict
    the probability of an instance being in a class. Outputted predictions
    will be a integer of 0 or 1 corresponding to the most likely class the
    classifier predicts the instance to be in. By default a instance is given
    a label of 1 if the probability of it lying in this class is > 0.5. If
    desired, that 0.5 threshold can be adjusted, depending on what kind of
    classifier is desired.

    Parameters
    ----------
    learning_rate : float
        A float that determines the learning rate that will be used to update
        the weights and biases of the model with gradient descent.
    num_iterations: int
        An integer value to determine how many steps of gradient descent will
        be used until the weights and biases are chosen.

    Attributes
    ----------
    learning_rate: float
        The learning rate as the class is initialised with.
    num_iterations: int
        The num of iterations as chosen when the class is initiated.
    X_: np.array
        A numpy array of the feature set. This is only assinged if the fit()
        method is called with a feature set.
    y_: np.array
        A numpy array with the target variables. This is also only assigned if
        the fit() method is called.
    b_: float
        The intercept term. In a single feature linear regression, this would
        be the 'c' in 'y = mx + c'.
    W_: np.array
        The weights assigned to the features in the feature set.
    costs_: list
        A list of the costs - in binary cross entropy, during each epoch of
        gradient descent as it optimizes.
    """

    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def _sigmoid_function(self, z):
        """Sigmoid function

        Used in Logistic Regression to correct the log odds values to a
        predicted probability, a float between 0 and 1.

        Parameters
        ----------
        z: float
            A log odds value. This can be anything between negative infinity
            and infinity.

        Returns
        -------
        sigmoid: float
            A probability value as transformed by the sigmoid value. Will be
            between 0 and 1.
        """
        sigmoid = 1.0 / (1.0 + np.exp(-(np.clip(z, -250, 250))))
        return sigmoid

    def predict_probabilities(self, X):
        """Function to predict the probability of an instance's class.

        Wraps the sigmoid function previously defined around the linear
        formula to get the predicted probabilities of an intsance being
        in a certain class. The output will be a numpy array of floats,
        between 0 and 1. Where 0 is a negative class and 1 is a positive class.

        Parameters
        ----------
        X : np.array
            A numpy array of the features for each instance.

        Returns
        -------
        y_pred: np.array
            A numpy array of floats between 0 and 1 with the predicted
            probabilities.
        """
        y_pred = self._sigmoid_function((X @ self.W_) + self.b_)
        return y_pred

    def predict(self, X, threshold=0.5):
        y_pred_probability = self.predict_probabilities(X)
        y_pred = np.where(y_pred_probability > 0.5, 1, 0)
        return y_pred

    def _cost_function(self, y_pred):
        """Calculates the cost function - binary cross entropy

        Parameters
        ----------
        y_pred: np.array
            A numpy array of the predicted probabilities.

        Returns
        -------
        cost: float
            A float of the cost function value.
        """
        cost = (
            -self.y_ @ (np.log(y_pred))
            - ((1 - self.y_) @ np.log(1 - y_pred)) / self.m_
        )
        return cost

    def _update_weights(self, y_pred):
        """Function to update the weights for the prediction function.

        Parameters
        ----------
        y_pred : np.array
            A numpy array of the predicted probabilities.
        """
        db = -2 * np.mean(self.y_ - y_pred)
        dW = -2 * ((self.X_.T @ (self.y_ - y_pred)) / self.n_)

        self.b_ = self.b_ - (self.learning_rate * db)
        self.W_ = self.W_ - (self.learning_rate * dW)

    def fit(self, X, y):
        """Method to fit the model using feature and targets set

        Parameters
        ----------
        X : np.array
            A numpy array containing the features
        y : np.array
            A numpy array containing the targets
        """
        self.X_ = X
        self.y_ = y

        # shape of the data
        self.n_ = self.X_.shape[0]
        self.m_ = self.X_.shape[1]

        # initialise paramters
        self.b_ = 0.0
        self.W_ = np.zeros(self.m_)

        # cost array
        self.costs_ = []

        for i in range(self.num_iterations):
            predictions = self.predict_probabilities(X)
            cost = self._cost_function(predictions)
            self.costs_.append(cost)
            self._update_weights(predictions)
