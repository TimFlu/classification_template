import numpy as np
import logging
logger = logging.getLogger(__name__)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def metric_evaluation(labels, pred, binary=True, num_classes=None):
    # For AUC calculation
    if binary:
        # Binary AUC
        if len(np.unique(labels)) < 2:
            auc = np.nan  # Handle missing classes
        else:
            auc = roc_auc_score(labels, pred)
        # Convert predictions to binary (0 or 1) for metrics
        pred = [1 if i > 0.5 else 0 for i in pred]
    else:
        # Multiclass AUC: Check if all classes are represented
        if len(np.unique(labels)) < num_classes:
            auc = np.nan  # Handle missing classes
        else:
            # One-hot encode labels and predictions for AUC
            labels_one_hot = np.eye(num_classes)[labels]
            pred_one_hot = np.eye(num_classes)[np.argmax(pred, axis=1)]
            auc = roc_auc_score(labels_one_hot, pred_one_hot, multi_class='ovr')

        # Convert predictions to class indices for metrics
        pred = np.argmax(pred, axis=1)
        
    # Accuracy
    accuracy = accuracy_score(labels, pred)

    # Precision, Recall, F1-Score
    if binary:
        # For binary classification, no averaging needed
        precision = precision_score(labels, pred, zero_division=0)
        recall = recall_score(labels, pred, zero_division=0)
        f1 = f1_score(labels, pred, zero_division=0)
    else:
        # For multiclass, use macro-averaging (or adjust as needed)
        precision = precision_score(labels, pred, average='macro', zero_division=0)
        recall = recall_score(labels, pred, average='macro', zero_division=0)
        f1 = f1_score(labels, pred, average='macro', zero_division=0)

    return accuracy, precision, recall, f1, auc

def expected_calibration_error(labels, y_pred, num_bins=10):
    y_pred = np.array(y_pred)
    labels = np.array(labels)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_pred_confidence = np.max(y_pred, axis=1)

    bins = np.linspace(0, 1, num_bins + 1) # Bin edges
    bin_counts = np.zeros(num_bins) # Number of predictions in each bin
    bin_correct = np.zeros(num_bins) # Number of correct predictions in each bin
    bin_confidence = np.zeros(num_bins) # Mean confidence in each bin

    for label, pred_conf, pred_class in zip(labels, y_pred_confidence, y_pred_class):
        bin_idx = np.digitize(pred_conf, bins) - 1
        if bin_idx >= num_bins: # Account for edge case
            bin_idx = num_bins - 1
        bin_counts[bin_idx] += 1
        bin_confidence[bin_idx] += pred_conf
        if pred_class == label:
            bin_correct[bin_idx] += 1

    bin_accuracy = np.nan_to_num(bin_correct / bin_counts)
    bin_confidence = np.nan_to_num(bin_confidence / bin_counts)

    ece = np.sum(bin_counts * np.abs(bin_accuracy - bin_confidence)) / len(labels)
    return ece

# **************** Early Stopping ***************** #
class EarlyStopping:
    """
    Early stopping during training to avoid overfitting

    Attributes:
    patience (int): how many times in row the early stopping condition was not fullfilled
    mind_delta (float): how big the relative loss between two consecutive trainings must be
    counter (int): how many consecutive times the stopper was triggered
    best_loss (float): the best loss achieved in the training so far, used to calculate the relative loss
    early_stop (bool): False if stopper did not reach the patience and the training should continue. True if training should end.

    Methods:
    __call__: Determines if the current loss is correcting the model enough or not and if the training should end or not.
    """
    def __init__(self, patience=5, min_delta=1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = 10e10
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Determines if the current loss is correcting the model enough or not and if the training should end or not.

        Input:
        val_loss: the current loss of the training
        """
        relative_loss = (self.best_loss - val_loss) / self.best_loss * 100
        logger.info(f"Early stopping relative loss = {relative_loss}")
        if relative_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif relative_loss < self.min_delta:
            self.counter += 1
            logger.info(
                f"Early stopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                logger.info("Early stopping")
                self.early_stop = True