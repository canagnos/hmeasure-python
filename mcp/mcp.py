from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


# Calculate raw ROC, replacing any tied sequences by a diagonal
# Raw ROC starts at F0[1]=0, F1[1]=0, and ends at F0[K1]=1, F1[K1]=1.
def get_score_distributions(true_labels, scores):

    n1 = np.sum(true_labels)
    n0 = np.sum(1-true_labels)
    n = n1 + n0

    # Count the instances of each unique score for label 1s, and ranks them by score
    df = pd.DataFrame([
        np.asarray(true_labels).reshape(n),
        np.asarray(1-true_labels).reshape(n),
        np.asarray(scores).reshape(n1 + n0)
    ]).transpose()
    df.columns = ['true_labels','true_labels_inverted', 'scores']

    s1 = np.asarray(df.groupby('scores')['true_labels'].sum()/n1)
    s0 = np.asarray(df.groupby('scores')['true_labels_inverted'].sum()/n0)
    ind = np.asarray(df['scores'])

    # Make sure to add the endpoints 0,0 and 1,1 for convenience even if they appear already
    s1 = np.concatenate(([0], s1, [1-np.sum(s1)]))
    s0 = np.concatenate(([0], s0, [1 - np.sum(s0)]))
    ind = np.concatenate(([0], ind, [1]))
    s = len(ind)
    f1 = np.cumsum(s1)
    f0 = np.cumsum(s0)

    return {'f1':f1, 'f0':f0, 's1':s1, 's0':s0, 's':s}


def plot_single_roc(true_labels, scores):
    out = get_score_distributions(true_labels, scores)
    plt.plot(out['f1'], out['f0'])
    return True


def h_measure_single(
        true_labels,
        scores,
        severity_ratio=0.5,
        threshold=0.5,
        level=0.5,
        verbose=True
):
    assert np.all(true_labels.shape == scores.shape)
    assert len(true_labels.shape) == 1 or true_labels.shape[1] == 1
    assert type(true_labels) is np.matrix or type(true_labels) is np.ndarray
    assert type(scores) is np.matrix or type(scores) is np.ndarray

    # this is a numeric version of the class labels
    n1 = np.sum(true_labels)
    n0 = np.sum(1-true_labels)
    n = n1 + n0
    pi0 = n0 / n
    pi1 = n1 / n

    # retrieve severity ratio - set to default if absent
    if severity_ratio is None:
        severity_ratio = pi1/pi0

    # order data into increasing scores
    scores_sorted = np.sort(scores)
    scores_order = np.argsort(scores)

    out = get_score_distributions(true_labels, scores)

    # now compute statistics - start with AUC
    auc = 1.0 - np.sum(out['s0'] * (out['f1'] - 0.5 * out['s1']))
    if auc < 0.5 and verbose:
        print 'ROC curve lying below the diagonal. Double-check scores.'

    return {'auc':auc}

    # move to scalar misclassification statistics
    conf_matrix = confusion_matrix(true_labels, scores > threshold)

    conf_matrix_metrics

    #misclass_out = misclass_counts((s>threshold),true_class)
    #misclass_metrics = misclass_out['metrics']
    #temp = misclass_out['conf_matrix']
    #misclass_conf = dataFrame(
    #	TP=temp[1,1], FP=temp[2,1],
    #	TN=temp[2,2], FN=temp[1,2])
    return True

def confusion_matrix_metrics(conf_matrix):

    # note the semantics of conf_matrix:
    TN = conf_matrix[0][0]
    FN = conf_matrix[1][0]



def h_measure(
        true_labels,
        scores,
        severity_ratio=None,
        threshold=None,
        level=None
):

    # let's say that trueLabels is 1xn and scores kxn np matrices
    assert true_labels.shape[1] == scores.shape[1]
    assert true_labels.shape[0] == 1
    assert type(true_labels) is np.matrix
    assert type(scores) is np.matrix

    # make sure the only two class labels are 0 and 1
    assert np.all(np.sort(np.unique(np.array(trueLabels)[0])) == np.array([0, 1]))

    # validate optional arguments
    assert 0 <= severity_ratio <= 1
    assert 0 <= level <= 1
    assert 0 <= threshold

    return True

