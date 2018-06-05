import pandas as pd
import numpy as np
import mcp.mcp as mcp
reload(mcp)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



true_labels = np.asarray([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
scores1 = np.asarray([0, 0.1, 0.2, 0.3, 0.3, 0.5, 0.6, 0.8, 0.8, 0.9])
scores2 = np.asarray([0.4, 0.1, 0.5, 0.2, 0.3, 0.0, 0.5, 0.2, 0.2, 0.15])
scores = scores1


mcp.get_score_distributions(true_labels, scores1) # ours
precision_recall_curve(true_labels, scores1) # fail
roc_curve(true_labels, scores1) # fail


mcp.plot_single_roc(true_labels, scores1)
mcp.plot_single_roc(true_labels, scores2)

our_auc = mcp.h_measure_single(true_labels, scores1)['auc']
sk_auc = roc_auc_score(true_labels, scores1) # pass
assert our_auc == sk_auc

