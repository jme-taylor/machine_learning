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


def true_positive(true, pred):
    """
    Returns the amount of true positives that a classifier has made.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred: list
        The predictions the classifier has made

    Returns
    -------
    tps: int
        The count of true positives from the classifier
    """
    tps = 0

    for t, p in zip(true, pred):
        if t == 1 and p == 1:
            tps += 1

    return tps


def false_positive(true, pred):
    """
    Returns the amount of false positives that a classifier has made.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred: list
        The predictions the classifier has made

    Returns
    -------
    fps: int
        The count of false positives from the classifier
    """
    fps = 0

    for t, p in zip(true, pred):
        if t == 0 and p == 1:
            fps += 1

    return fps


def false_negative(true, pred):
    """
    Returns the amount of false negatives that a classifier has made.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred: list
        The predictions the classifier has made

    Returns
    -------
    fns: int
        The count of false negatives from the classifier
    """
    fns = 0

    for t, p in zip(true, pred):
        if t == 1 and p == 0:
            fns += 1

    return fns


def true_negative(true, pred):
    """
    Returns the amount of true negatives that a classifier has made.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred: list
        The predictions the classifier has made

    Returns
    -------
    tns: int
        The count of true negatives from the classifier
    """
    tns = 0

    for t, p in zip(true, pred):
        if t == 0 and p == 0:
            tns += 1

    return tns


def precision(true, pred):
    """
    Gets the precision score of a classifier, as defined by the amount of true
    positives, divided by the sum of true positives and false positives (i.e.
    all the predicted positive values). Precision can be any value between 0
    and 1, with 1 being best.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred: list
        The predicted labels from the classifier

    Returns
    -------
    precision: float
        The precision score, a float between 0 and 1.
    """
    tps = true_positive(true, pred)
    fps = false_positive(true, pred)

    precision = tps / (tps + fps)

    return precision


def recall(true, pred):
    """
    Gets the recall score of a classifier, as defined by the amount of true
    positives, divided by the sum of true positives and false negatives (i.e.
    all the true positive values). Recall can be any value between 0 and 1,
    with 1 being best.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred: list
        The predicted labels from the classifier

    Returns
    -------
    recall: float
        The recall score, a float between 0 and 1.
    """
    tps = true_positive(true, pred)
    fns = false_negative(true, pred)

    recall = tps / (tps + fns)

    return recall
