import numpy as np
import pandas as pd
from typing import List
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error



def norm(x: float) -> float:
    """Normalize vector"""
    return (x - np.min(x)) / np.ptp(x)

def wide_to_long(wide: np.array, possible_ratings: List[int]) -> np.array:
    """Go from wide table to long.
    :param wide: wide array with user-item interactions
    :param possible_ratings: list of possible ratings that we may have."""

    def _get_ratings(arr: np.array, rating: int) -> np.array:
        """Generate long array for the rating provided
        :param arr: wide array with user-item interactions
        :param rating: the rating that we are interested"""
        idx = np.where(arr == rating)
        return np.vstack(
            (idx[0], idx[1], np.ones(idx[0].size, dtype="int8") * rating)
        ).T

    long_arrays = []
    for r in possible_ratings:
        long_arrays.append(_get_ratings(wide, r))

    return np.vstack(long_arrays)


def make_tf_dataset(
    df: pd.DataFrame,
    targets: List[str],
    val_split: float = 0.1,
    batch_size: int = 512,
    seed=42,
):
    """
    Make TensorFlow dataset from Pandas DataFrame.
        - param df: input DataFrame - only contains features and target(s)
        - param targets: list of columns names corresponding to targets
        - param val_split: fraction of the data that should be used for validation
        - param batch_size: batch size for training
        - param seed: random seed for shuffling data - `None` won't shuffle the data
    """

    n_val = round(df.shape[0] * val_split)
    if seed:
        # shuffle all the rows
        x = df.sample(frac=1, random_state=seed).to_dict("series")
    else:
        x = df.to_dict("series")
    y = dict()
    for t in targets:
        y[t] = x.pop(t)
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    ds_val = ds.take(n_val).batch(batch_size)
    ds_train = ds.skip(n_val).batch(batch_size)
    return ds_train, ds_val


def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name} Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

import matplotlib.pyplot as plt
"""   
def precision_recall(recall_ncf, precision_ncf,recall_lightfm, precision_lightfm):
    plt.figure(figsize=(8, 8))
    plt.step(recall_ncf, precision_ncf, color='b', alpha=0.7, where='post', label='NCF')
    plt.step(recall_lightfm, precision_lightfm, color='r', alpha=0.7, where='post', label='LightFM')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
def ROC_curve(fpr_ncf,tpr_ncf,fpr_lightfm,tpr_lightfm):    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_ncf, tpr_ncf, color='b', alpha=0.7, label='NCF')
    plt.plot(fpr_lightfm, tpr_lightfm, color='r', alpha=0.7, label='LightFM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
"""
import matplotlib.pyplot as plt

def precision_recall(recall_ncf, precision_ncf, recall_lightfm, precision_lightfm):
    plt.figure(figsize=(10, 6))
    
    # Plot Precision-Recall curves
    plt.step(recall_ncf, precision_ncf, color='b', alpha=0.7, where='post', label='NCF')
    plt.step(recall_lightfm, precision_lightfm, color='r', alpha=0.7, where='post', label='LightFM')
    
    # Customize the plot
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def ROC_curve(fpr_ncf, tpr_ncf, fpr_lightfm, tpr_lightfm):    
    plt.figure(figsize=(10, 6))
    
    # Plot ROC curves
    plt.plot(fpr_ncf, tpr_ncf, color='b', alpha=0.7, label='NCF')
    plt.plot(fpr_lightfm, tpr_lightfm, color='r', alpha=0.7, label='LightFM')
    
    # Customize the plot
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Example usage:
# beautify_precision_recall(recall_ncf, precision_ncf, recall_lightfm, precision_lightfm)
# beautify_ROC_curve(fpr_ncf, tpr_ncf, fpr_lightfm, tpr_lightfm)
