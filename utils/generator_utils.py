import random

class EarlyStopping:
    """
    Implements early stopping to terminate training when a monitored metric has stopped improving.

    This class is typically used during model training to prevent overfitting and reduce training time by 
    stopping the training process if no improvement is seen over a specified number of epochs.

    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        mode (str): 'min' if a lower metric is better (e.g., loss), 'max' if a higher metric is better (e.g., accuracy).
        counter (int): Number of consecutive epochs with no improvement.
        best_score (float or None): Best metric score seen so far.
        early_stop (bool): Whether early stopping should be triggered.
        monitor_op (Callable): Function used to compare metric values based on mode and min_delta.
    """
    
    def __init__(self, patience :int=5, min_delta: float=0, mode: str='min'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement in validation loss.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): Whether to 'min'imize or 'max'imize the metric ('min' for loss, 'max' for accuracy, etc.).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        if self.mode == 'min':
            self.monitor_op = lambda a, b: a < b - self.min_delta
        else:  # 'max' mode
            self.monitor_op = lambda a, b: a > b + self.min_delta

    def check_early_stop(self, current_score: float=None):
        """
        Checks if early stopping should be triggered based on the current score.
        
        Args:
            current_score (float): Current metric value (e.g., validation loss or accuracy).
        
        Returns:
            None: Updates internal state for early stopping.
        """
        if self.best_score is None:
            self.best_score = current_score
        elif self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def augment_text(text):
    """
    Performs simple text augmentation by randomly reversing a subset of words in the input string.

    This method is useful for introducing noise or variation in text data, which can help improve 
    model robustness during training.

    Args:
        text (str): The input string to be augmented.

    Returns:
        str: A new string with a few randomly chosen words reversed in place.
    
    Notes:
        - The number of words to reverse is randomly chosen between 1 and 10% of the total words (at least 1).
        - Each selected word is replaced by its reversed form.
    """
    words = text.split()
    num_replacements = random.randint(1, max(1, len(words) // 10))
    for _ in range(num_replacements):
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx][::-1]  # <-- reverse the word
    return " ".join(words)

