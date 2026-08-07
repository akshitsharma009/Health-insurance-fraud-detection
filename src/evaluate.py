"""
evaluate.py
-----------
Reusable evaluation utilities for the fraud detection model.

Previously this file was empty -- evaluation logic lived only inline in
notebooks/03_modeling.ipynb, which made it impossible to re-run a
consistent evaluation outside of a notebook (e.g. from a CI script when
retraining on new data). This module extracts that logic into reusable
functions.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_at_threshold(y_true, y_prob, threshold=0.3):
    """
    Compute precision, recall, and F1 for fraud (positive class) at a
    given decision threshold.
    """

    preds = (np.asarray(y_prob) >= threshold).astype(int)

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds),
    }


def sweep_thresholds(y_true, y_prob, thresholds=None):
    """
    Evaluate precision/recall/F1 across a range of thresholds, useful for
    picking an operating point. Returns a list of dicts, one per threshold.
    """

    if thresholds is None:
        thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    return [evaluate_at_threshold(y_true, y_prob, th) for th in thresholds]


def full_report(y_true, y_prob, threshold=0.3):
    """
    Print a full evaluation report: ROC-AUC, classification report, and
    confusion matrix, at a given decision threshold.
    """

    preds = (np.asarray(y_prob) >= threshold).astype(int)

    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"\nClassification report (threshold={threshold}):")
    print(classification_report(y_true, preds, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, preds))
