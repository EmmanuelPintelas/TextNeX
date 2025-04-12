import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
    confusion_matrix
    )

def run_evaluation(
        predictions: np.array=None,
        probabilities: np.array=None,
        labels_query: np.array=None
        ) -> dict[str, list]:
    """
    Evaluate classification performance using various metrics.

    This function computes several evaluation metrics based on the predicted labels,
    predicted probabilities, and the true labels. The computed metrics include
    Accuracy (Acc), F1 score (F1), Precision (Pre), Area Under the ROC Curve (AUC),
    Geometric Mean (GM), and the confusion matrix (cm). These are stored in a dictionary
    with each metric wrapped in a list.

    Args:
        predictions (array-like): Predicted class labels.
        probabilities (array-like): Predicted probabilities for the positive class.
        labels_query (array-like): True class labels.

    Returns:
        dict: A dictionary containing evaluation metrics, where each key maps to a
              list containing the corresponding metric.
              Keys: 'Acc', 'F1', 'Pre', 'AUC', 'GM', 'cm'.
    """
    evaluation_metrics = {'Acc': [], 'F1': [], 'Pre': [], 'AUC': [], 'GM': [], 'cm': []}
    acc, f1, pre, auc, gm, cm = scores(predictions, probabilities, labels_query)
    evaluation_metrics['Acc'].append(acc)
    evaluation_metrics['F1'].append(f1)
    evaluation_metrics['Pre'].append(pre)
    evaluation_metrics['AUC'].append(auc)
    evaluation_metrics['GM'].append(gm)
    evaluation_metrics['cm'].append(cm)
    return evaluation_metrics

def scores(
        predictions: np.array=None,
        probabilities: np.array=None,
        labels_query: np.array=None
        ) -> tuple:
    """
    Calculate various performance metrics for classification results.
    
    This function computes accuracy, F1 score, precision, ROC AUC, geometric mean of recall,
    and confusion matrix based on the provided predictions, probability scores, and true labels.
    
    Args:
        predictions (array-like): The predicted class labels.
        probabilities (array-like): The probability estimates or confidence scores for each class.
            For binary classification, this should be a 1D array of probabilities for the positive class.
            For multiclass classification, this should be a 2D array with shape (n_samples, n_classes).
        labels_query (array-like): The true class labels.
    
    Returns:
        tuple: A tuple containing:
            - acc (float): Accuracy score, rounded to 4 decimal places.
            - f1 (float): Weighted F1 score, rounded to 4 decimal places.
            - pre (float): Weighted precision score, rounded to 4 decimal places.
            - auc (float): Weighted ROC AUC score, rounded to 4 decimal places.
            - gm (float): Geometric mean of class-wise recalls, rounded to 4 decimal places.
            - cm (array): Confusion matrix.
    
    Notes:
        - For multiclass problems (more than 2 classes), ROC AUC is calculated with 'ovr' (one-vs-rest) strategy.
        - Precision calculation handles zero division by setting the result to 0.
        - Geometric mean is calculated using the recall of each class.
    """
    acc = accuracy_score(labels_query, predictions)
    f1 = f1_score(labels_query, predictions, average='weighted')
    pre = precision_score(labels_query, predictions, average='weighted', zero_division=0)
    if len(set(labels_query)) > 2:
        auc = roc_auc_score(labels_query, probabilities, multi_class='ovr', average='weighted')
    else:
        auc = roc_auc_score(labels_query, probabilities, average='weighted')
    cm = confusion_matrix(labels_query, predictions)
    recalls_per_class = np.diag(cm) / np.sum(cm, axis=1)
    gm = np.prod(recalls_per_class) ** (1.0 / len(recalls_per_class))
    return round(acc, 4), round(f1, 4), round(pre, 4), round(auc, 4), round(gm, 4), cm


class TextDataset(Dataset):
    """
    A PyTorch Dataset for text classification tasks.
    
    This dataset handles the preprocessing of text data using a tokenizer
    and converts texts and labels into a format suitable for training models.
    
    Attributes:
        texts (list): List of text strings to be processed.
        labels (list): List of labels corresponding to each text.
        tokenizer: The tokenizer used to convert text to tokens.
        max_length (int): Maximum sequence length for padding/truncation.
    """
    def __init__(
            self,
            texts: list[str]=None,
            labels: np.array=None,
            tokenizer: AutoTokenizer=None,
            max_length: int=128
            ):
        """
        Initialize the TextDataset.
        
        Args:
            texts (list): List of text strings to be processed.
            labels (list): List of labels corresponding to each text.
            tokenizer: The tokenizer used to convert text to tokens.
            max_length (int, optional): Maximum sequence length. Defaults to 128.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: The number of samples.
        """
        return len(self.texts)
    
    def __getitem__(self, idx: int=None) -> torch.tensor:
        """
        Get a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            tuple: A tuple containing:
                - input_ids tensor: Token IDs for the text
                - attention_mask tensor: Attention mask for the tokens
                - label tensor: The corresponding label
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return encoding.input_ids.squeeze(), encoding.attention_mask.squeeze(), torch.tensor(label)
