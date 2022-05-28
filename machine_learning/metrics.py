import numpy as np


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


def f1_score(true, pred):
    """
    Function to get the F1 score of a classifier, which is the harmonic mean
    of the recall and the precision. It can score 0 at worst and 1 at best.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred: list
        The predicted labels from the classifier

    Returns
    -------
    f1_score: float
        The f1 score, a float between 0 and 1.
    """

    p = precision(true, pred)
    r = recall(true, pred)

    f1_score = 2 * p * r / (p + r)

    return f1_score


def false_positive_rate(true, pred):
    """
    Function to record the false positive rate (FPR) of a classifier.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred: list
        The predicted labels from the classifier

    Returns
    -------
    fpr: float
        The f1 score, a float between 0 and 1.
    """
    fp = false_positive(true, pred)
    tn = true_negative(true, pred)
    fpr = fp / (fp + tn)

    return fpr


def _fpr_tpr_scores(true, pred_probabilities, num_thresholds=101):
    """
    Function to get a set of true positive rates (TPR) and false positive
    rates (FPR) given a certain threshold of a classifier.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred_probabilities: list
        A list of predicitions from the classifier, in float format, as their
        raw probabilities.

    num_thresholds: int, optional
        The number of thresholds, evenly spaced between 1 and 0 that you'd
        like to use. Defaults at 101.

    Returns
    -------
    tpr_list: list
        A list of true positive ratio values

    fpr_list: list
        A list of false postitive ratio values
    """
    # use arbitrary list of thresholds if not given

    thresholds = np.linspace(0, 1, num_thresholds)

    tpr_list = []
    fpr_list = []

    for t in thresholds:
        temp_pred = [1 if x >= t else 0 for x in pred_probabilities]
        tpr = recall(true, temp_pred)
        tpr_list.append(tpr)
        fpr = false_positive_rate(true, temp_pred)
        fpr_list.append(fpr)

    return tpr_list, fpr_list


def area_under_roc_curve(true, pred_probabilities, num_thresholds=101):
    """
    Function to score the area under the receiver operating characteristic
    (ROC) curve. Scores can range between 0 and 1. 1 is a perfect classifer,
    0.5 is a purely random classifier, and 0 is very bad.

    Parameters
    ----------
    true: list
        The ground truth labels

    pred_probabilities: list
        A list of predicitions from the classifier, in float format, as their
        raw probabilities.

    num_thresholds: int, optional
        The number of thresholds, evenly spaced between 1 and 0 that you'd
        like to use. Defaults at 101.

    Returns
    -------
    area_under_roc_curve: float
        A float with the area under the roc curve score
    """
    tpr_list, fpr_list = _fpr_tpr_scores(
        true, pred_probabilities, num_thresholds
    )

    area_under_roc_curve = np.trapz(y=tpr_list, x=fpr_list) * -1

    return area_under_roc_curve


def plot_roc_curve(
    true, pred_probabilities, axes, show_area_score=True, num_thresholds=101
):
    """
    Function to plot the receiver operating characteristic (ROC) curve. This
    is the false positive rate (FPR) plotted against the true positive rate
    (TPR).

    Parameters
    ---------
    true: list
        The ground truth labels

    pred_probabilities: list
        A list of predicitions from the classifier, in float format, as their
        raw probabilities.

    axes: matplotlib.Axes
        A matplotlib subplot to do the plotting upon

    num_thresholds: int, optional
        The number of thresholds, evenly spaced between 1 and 0 that you'd
        like to use. Defaults at 101.

    Returns
    -------
    axes: matplotlib.Axes
        An ammended matplotlib subplot containing the plot of the ROC curve
    """

    fprs, tprs = _fpr_tpr_scores(true, pred_probabilities, num_thresholds)

    print(len(fprs))

    if show_area_score:
        aurc = area_under_roc_curve(true, pred_probabilities)
        title = f"Receiver operating characteristic curve - {aurc:.2f} area under score"  # noqa
    else:
        title = "Receiver operating characteristic curve"

    axes.plot(fprs, tprs)
    axes.fill_between(fprs, tprs, alpha=0.2)
    axes.set(
        title=title,
        xlabel="FPR",
        ylabel="TPR",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    return axes
