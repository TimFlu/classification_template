import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_confidence_histogram(y_true, y_pred, save_path, num_bins=10,):
    # Calculate accuracy
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_true == y_pred_class)
    mean_confidence = np.mean(np.max(y_pred, axis=1))

    fig = plt.figure()
    plt.hist(np.max(y_pred, axis=1), bins=num_bins, edgecolor='black', alpha=0.7) 
    plt.vlines(accuracy, 0, plt.gca().get_ylim()[1], colors='r', linestyles='dashed', label=f"Accuracy: {accuracy:.2f}")
    plt.vlines(mean_confidence, 0, plt.gca().get_ylim()[1], colors='g', linestyles='dashed', label=f"Mean confidence: {mean_confidence:.2f}")
    plt.title("Confidence histogram")
    plt.xlabel("Confidence")
    plt.ylabel("samples")
    plt.legend()
    plt.savefig(f"{save_path}/confidence_histogram.png")
    return fig

def plot_reliability_diagram(y_true, y_pred, save_path, num_bins=10):
    # Calculate accuracy
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_pred_confidence = np.max(y_pred, axis=1)
    
    bins = np.linspace(0, 1, num_bins + 1) # Bin edges
    bin_centers = (bins[:-1] + bins[1:]) / 2 # Bin centers
    bin_counts = np.zeros(num_bins) # Number of predictions in each bin
    bin_correct = np.zeros(num_bins) # Number of correct predictions in each bin
    bin_confidence = np.zeros(num_bins) # Mean confidence in each bin

    for label, pred_conf, pred_class in zip(y_true, y_pred_confidence, y_pred_class):
        bin_idx = np.digitize(pred_conf, bins) - 1
        if bin_idx >= num_bins: # Account for edge case
            bin_idx = num_bins - 1
        bin_counts[bin_idx] += 1
        bin_confidence[bin_idx] += pred_conf
        if pred_class == label:
            bin_correct[bin_idx] += 1

    bin_accuracy = np.nan_to_num(bin_correct / bin_counts)
    bin_confidence = np.nan_to_num(bin_confidence / bin_counts)

    fig = plt.figure()
    plt.bar(bin_centers, bin_accuracy, width=1/num_bins, label='Accuracy', alpha=0.7)
    for acc, diag in zip(bin_accuracy, bin_centers):
        if acc < diag:
            plt.bar(diag, diag-acc, bottom=acc, width=1/num_bins, color='red', alpha=0.2, hatch='/')
        else:
            plt.bar(diag, acc-diag, bottom=diag, width=1/num_bins, color='red', alpha=0.2, hatch='/')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    plt.title("Reliability diagram")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{save_path}/reliability_diagram.png")

    return fig

    


if __name__ == "__main__":
    # Example: Multiclass predictions
    y_true = np.array([1, 1, 0, 1, 2, 2, 1, 0, 2, 1])  # True labels
    y_pred = np.array([
        [0.1, 0.2, 0.1],  # Predicted probabilities for sample 1
        [0.1, 0.6, 0.3],
        [0.2, 0.2, 0.6],
        [0.1, 0.8, 0.1],
        [0.9, 0.05, 0.05],
        [0.2, 0.3, 0.5],
        [0.15, 0.7, 0.15],
        [0.2, 0.1, 0.1],
        [0.1, 0.3, 0.2],
        [0.2, 0.7, 0.1]
    ])

    # Plot confidence histogram
    plot_confidence_histogram(y_true, y_pred, num_bins=10, save_path=os.getcwd())
    plot_reliability_diagram(y_true, y_pred, save_path=os.getcwd(), num_bins=10)