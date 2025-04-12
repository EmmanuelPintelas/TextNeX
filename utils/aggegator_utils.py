import torch
import numpy as np
from scipy.optimize import minimize
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils.utils import TextDataset, scores, run_evaluation

def get_val_loader(
        tokenizer: AutoTokenizer=None,
        val_texts: list[str]=None,
        val_labels: np.array=None
        ) -> DataLoader:
    """
    Create and return a DataLoader for validation data.
    
    This function initializes a TextDataset with validation texts and labels,
    then wraps it in a PyTorch DataLoader for batch processing during validation.
    
    Args:
        tokenizer: The tokenizer to use for processing the validation texts.
            Should be compatible with the TextDataset class requirements.
    
    Returns:
        DataLoader: A PyTorch DataLoader containing the validation dataset
            with specified batch size.
    
    Notes:
        - Uses a fixed maximum sequence length of 106 tokens
        - Uses a batch size of 256 for validation
        - Assumes val_texts and val_labels are defined in the global scope
    """
    max_length = 106
    batch_size = 256
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return val_loader


def extract_model_outputs(
        models: list=None,
        tokenizers: list=None,
        device: str=None
        ) -> list[torch.tensor]:
    """
    Extract prediction logits from multiple models using their respective tokenizers.
    
    This function evaluates multiple models on the same validation dataset,
    collecting the output logits from each model. It processes the validation data
    in batches and returns the concatenated logits for each model.
    
    Args:
        models (list): List of PyTorch models to evaluate.
        tokenizers (list): List of tokenizers corresponding to each model.
            Each tokenizer is used to process the text data for its respective model.
        device (str): Device used for PyTorch inference (`cpu` or `cuda`)
            
    Returns:
        list: A list of tensors, where each tensor contains the logits from one model
            for the entire validation dataset. The shape of each tensor is
            [num_samples, num_classes].
    
    Notes:
        - Models are evaluated in evaluation mode (model.eval())
        - Computation is performed with torch.no_grad() for memory efficiency
        - Input data is moved to the device specified by the global 'device' variable
        - All logits are returned as CPU tensors for consistency
    """
    logits_list = []
    for model, tokenizer in zip(models, tokenizers):
        model.eval()
        val_loader = get_val_loader(tokenizer)
        all_logits = []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
        logits_list.append(torch.cat(all_logits, dim=0))
    
    return logits_list


def ensemble_loss(weights, logits_list, labels, best_gm, best_weights):
    """
    Compute negative G-mean loss for ensemble weight optimization.

    This function computes the weighted ensemble logits using the provided weights,
    applies softmax to obtain class probabilities, and evaluates predictions using
    the G-mean metric. It returns the negative G-mean (to be minimized), along with
    the best G-mean and corresponding best weights.

    Args:
        weights (list or np.array): Ensemble weights for each model.
        logits_list (list of torch.Tensor): List of model logits, each of shape [batch_size, num_classes].
        labels (np.array): True class labels for the batch.
        best_gm (float): Best G-mean score observed so far.
        best_weights (np.array): Ensemble weights that produced the best G-mean.

    Returns:
        tuple:
            - float: Negative G-mean (for minimization in optimizers).
            - float: Updated best G-mean.
            - np.array: Updated best weights.
    """
    weights = np.array(weights)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    stacked_logits = torch.stack(logits_list)  # Shape: [n_models, batch_size, n_classes]
    weighted_logits = torch.sum(weights_tensor * stacked_logits, dim=0)

    probs = torch.softmax(weighted_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    prob_positives = probs[:, 1] if probs.shape[1] == 2 else probs
    _, _, _, _, gm, _ = scores(preds, prob_positives, labels)

    if gm > best_gm:
        print(f"\n✅ New Best GM: {gm:.4f}")
        return -gm, gm, weights.copy()
    return -gm, best_gm, best_weights

class EnsembleOptimizer:
    """
    Optimizes ensemble weights to maximize G-mean based on model logits and true labels.

    This class encapsulates the optimization process for finding the best set of ensemble
    weights that maximize the geometric mean (G-mean) of predictions from multiple models.

    Attributes:
        logits_list (List[torch.Tensor]): List of logits from each model. Each tensor must have shape [batch_size, num_classes].
        labels (np.ndarray): Ground-truth class labels, assumed to be shape [batch_size].
        best_gm (float): Best G-mean achieved during optimization.
        best_weights (Optional[np.ndarray]): Weights corresponding to the best G-mean.
    """

    def __init__(self, logits_list: list[torch.Tensor], labels: np.array):
        """
        Initialize the EnsembleOptimizer.

        Args:
            logits_list (List[torch.Tensor]): List of model logits with shape [batch_size, num_classes] each.
            labels (np.ndarray): Ground-truth labels for evaluation.
        """
        self.logits_list = logits_list
        self.labels = labels
        self.best_gm = -float('inf')
        self.best_weights = None

    def loss(self, weights: list[float]) -> float:
        """
        Compute the loss (negative G-mean) for a given set of weights.

        This method is designed to be used with optimization routines that minimize a scalar function.

        Args:
            weights (List[float]): Ensemble weights to apply to each model's logits.

        Returns:
            float: Negative G-mean, which optimization will seek to minimize.
        """
        loss_val, updated_gm, updated_weights = ensemble_loss(
            weights,
            self.logits_list,
            self.labels,
            self.best_gm,
            self.best_weights
        )
        if updated_gm > self.best_gm:
            self.best_gm = updated_gm
            self.best_weights = updated_weights
        return loss_val

def SQP(logits_list: list[torch.Tensor], labels: np.array) -> tuple[np.array, float]:
    """
    Perform Sequential Quadratic Programming (SQP) to optimize ensemble weights for G-mean.

    This function uses Powell's method with constraints to find a set of weights that,
    when applied to model logits, maximize the geometric mean (G-mean) of predictions.

    Args:
        logits_list (List[torch.Tensor]): List of logits from each model, shape [batch_size, num_classes] per tensor.
        labels (np.ndarray): Ground-truth labels for evaluation, shape [batch_size].

    Returns:
        Tuple[np.ndarray, float]: 
            - Optimal weights that maximize G-mean.
            - Best G-mean achieved.

    Raises:
        ValueError: If optimization fails to improve G-mean.
    """
    optimizer = EnsembleOptimizer(logits_list, labels)

    num_models = len(logits_list)
    initial_weights = np.ones(num_models) / num_models
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(num_models)]

    minimize(
        optimizer.loss,
        initial_weights,
        method='Powell',
        bounds=bounds,
        constraints=constraints
    )

    if optimizer.best_weights is not None:
        return optimizer.best_weights, optimizer.best_gm
    else:
        raise ValueError("Optimization failed to improve GM")


def get_test_loader(
        tokenizer: AutoTokenizer=None,
        test_texts: list[str]=None,
        test_labels: np.array=None
        ) -> DataLoader:
    """
    Create and return a DataLoader for test data.
    
    This function initializes a TextDataset with test texts and labels,
    then wraps it in a PyTorch DataLoader for batch processing during testing
    or final evaluation.
    
    Args:
        tokenizer: The tokenizer to use for processing the test texts.
            Should be compatible with the TextDataset class requirements.
    
    Returns:
        DataLoader: A PyTorch DataLoader containing the test dataset
            with specified batch size.
    
    Notes:
        - Uses a fixed maximum sequence length of 106 tokens
        - Uses a batch size of 256 for testing
        - Assumes test_texts and test_labels are defined in the global scope
    """
    max_length = 106
    batch_size = 256
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader


def test_ensemble(
        models: list=None,
        weights: torch.tensor=None,
        tokenizer: AutoTokenizer=None,
        device: str=None,
        test_texts: list[str]=None,
        test_labels: np.array=None
        ):
    """
    Evaluate an ensemble of models on the test dataset.
    
    This function applies the specified weights to combine predictions from multiple models,
    processes the test dataset, and evaluates the ensemble's performance. It calculates
    and displays various performance metrics including accuracy, F1 score, precision,
    AUC, geometric mean, and confusion matrix.
    
    Args:
        models (list): A list of PyTorch models to include in the ensemble.
        weights (array-like): The weights to apply to each model's predictions.
            Should have the same length as the models list.
        tokenizer: The tokenizer to use for processing the test texts.
    
    Returns:
        None: Results are printed to the console.
    
    Notes:
        - Models must be in evaluation mode (model.eval()) before being passed to this function
        - All models must be on the same device (specified by the global 'device' variable)
        - The function assumes binary classification for probability extraction
          (using index 1 for positive class probabilities)
        - The results include detailed metrics and a confusion matrix
    """
    test_loader = get_test_loader(tokenizer, test_texts, test_labels)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).unsqueeze(2).to(device)
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits_list = [model(input_ids=input_ids, attention_mask=attention_mask).logits for model in models]
            stacked_logits = torch.stack(logits_list)  # [n_models, batch, n_classes]
            weighted_logits = torch.sum(weights_tensor * stacked_logits, dim=0)
            probs = torch.softmax(weighted_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Assuming binary classification
            all_labels.extend(labels.cpu().numpy())
    return scores(np.array(all_preds), np.array(all_probs), np.array(all_labels))