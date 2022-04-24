def accuracy(true, pred):
    """
    Returns the accuracy of a classification model, as defined as the amount
    of correct predictions, divided by the total amount of predictions

    Parameters
    ----------
    true: list
        The ground truth labels for each sample

    pred: list
        The predicted labels to calculate accuracy upon

    Returns
    -------
    accuracy: float
        A float, between 0 and 1, with the accuracy metric
    """
    correct = 0

    for t, p in zip(true, pred):
        if t == p:
            correct += 1

    accuracy = correct / len(true)
    return accuracy
