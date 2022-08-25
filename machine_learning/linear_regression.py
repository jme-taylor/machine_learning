import numpy as np


class LinearRegression:
    """Linear Regression using Gradient Descent to find optimal parameters.
    The loss function used is the mean squared error.

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
    X: np.array
        A numpy array of the feature set. This is only assinged if the fit()
        method is called with a feature set.
    y: np.array
        A numpy array with the target variables. This is also only assigned if
        the fit() method is called.
    b: float
        The intercept term. In a single feature linear regression, this would
        be the 'c' in 'y = mx + c'.
    W: np.array
        The weights assigned to the features in the feature set.
    costs: list
        A list of the costs (MSE) during each epoch of gradient descent as it
        optimizes.
    """

    def __init__(
        self,
        learning_rate,
        num_iterations,
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def predict(self, X):
        """Method that predicts a target value for a given set of features.

        This is done by multiplying the feature set by the weights of the
        model and adding the intercept term.

        Parameters
        ----------
        X: np.array
            A set of features, and their values

        Returns
        -------
        y_pred: np.array
            A set of predictions for the target variable
        """
        y_pred = X @ self.W + self.b
        return y_pred

    def get_residuals(self, y_pred):
        """Method to get the residuals of the predictions against true values.

        A residual is defined as y_true - y_pred. This is calculated across
        all values in the predictions.

        Parameters
        ----------
        y_pred : np.array
            An array of predicted values.

        Returns
        -------
        residuals: np.array
            A numpy array of the residuals of the predictions.
        """
        residuals = self.y - y_pred
        return residuals

    def get_cost_function(self, residuals):
        """Method to get the mean squared error of a prediction from residuals.

        This method uses the residuals, then squares them to get the sum of
        the squared residuals. This is then averaged by diving by the number
        of instances to get the mean squared error.

        Parameters
        ----------
        residuals : np.array
            A numpy array of the residuals.

        Returns
        -------
        cost: float
            A float of the MSE of the predictions.
        """
        cost = np.sum(np.square(residuals)) / self.n
        return cost

    def update_weights(self, residuals):
        """Method to update the weights of the model parameters.

        This is done using gradient descent. For both the weights and bias
        terms, the partial derivative, with respect to the cost function is
        taken. This partial derivative is then multiplied by the learning rate
        to get the change in parameter required. Finally, the weights are
        updated by taking the change in parameter from the original parameter.

        Parameters
        ----------
        residuals : np.array
            A numpy array of the residuals of the predictions.
        """
        db = -2 * (np.mean(residuals))
        dW = -2 * (self.X.T @ residuals / self.n)

        self.b = self.b - (self.learning_rate * db)
        self.W = self.W - (self.learning_rate * dW)

    def fit(self, X, y):
        """Method to fit a linear regression model to features and targets.

        The weights and biases of the model are initialised as zeros. For each
        epoch as defined by the num_iterations parameter of the class - the
        following occurs:
        - predictions are made to get the residuals
        - cost (MSE) is calculated and stored
        - the residuals are used to update the weights and bias terms

        Parameters
        ----------
        X : np.array
            A numpy array containing the feature set.
        y : np.array
            A numpy array containing the target variables.
        """
        self.X = X
        self.y = y

        # shape of the data - this will raise an index error if the data only
        #  has 1 feature
        self.n = self.X.shape[0]
        self.m = self.X.shape[1]

        self.b = 0.0
        self.W = np.zeros(self.m)

        self.costs = []

        for i in range(self.num_iterations):
            predictions = self.predict(X)
            residuals = self.get_residuals(predictions)
            cost = self.get_cost_function(residuals)
            self.costs.append(cost)
            self.update_weights(residuals)
