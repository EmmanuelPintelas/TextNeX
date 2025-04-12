import os
import umap
import torch
import numpy as np
from utils.utils import TextDataset
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, silhouette_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def data_loaders(
        tokenizer: object=None,
        train_texts: list[str]=None,
        train_labels: np.array=None,
        val_texts: list[str]=None,
        val_labels: np.array=None,
        test_texts: list[str]=None,
        test_labels: np.array=None
        ) -> tuple:
    """
    Creates PyTorch DataLoaders for training, validation, and test datasets using a given tokenizer.

    This function wraps preprocessed text and labels into dataset objects, then creates DataLoaders 
    for each dataset for use in model training and evaluation.

    Args:
        tokenizer: A tokenizer (e.g., from Hugging Face Transformers) used to tokenize the text inputs.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
            - test_loader (DataLoader): DataLoader for the test dataset.
    
    Notes:
        - Assumes `train_texts`, `train_labels`, `val_texts`, `val_labels`, `test_texts`, and `test_labels`
          are defined in the global scope.
        - Uses a fixed `max_length` of 106 and batch size of 256.
    """
    max_length = 106
    batch_size = 256
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def predictor(
        model: object=None,
        dataloader: DataLoader=None,
        device: str=None
        ) -> tuple:
    """
    Makes predictions using the provided model on the given DataLoader.

    This function evaluates the model in inference mode, processes the data in batches, 
    and returns predictions along with the associated probabilities.

    Args:
        model (torch.nn.Module): The model to be used for making predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the input data (e.g., validation or test data).
        device (str): Specify inference device (`cuda` or `cpu`)
        
    Returns:
        tuple: A tuple containing:
            - all_preds (np.ndarray): Array of predicted labels for each sample.
            - all_probs (np.ndarray): Array of probabilities for the positive class (index 1).
            - all_labels (np.ndarray): Array of ground truth labels for each sample.
    
    Notes:
        - This function uses `torch.no_grad()` to disable gradient computation for inference, improving efficiency.
        - Assumes a binary classification problem, where probabilities for the positive class are extracted using `[:, 1]`.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)[:, 1], np.array(all_labels)

def model_loader(
        model_file: str = None,
        model_load_path: str = None,
        device: str = None
        ) -> tuple:
    """
    Loads a pre-trained model and tokenizer from a specified file.

    This function attempts to load a model and tokenizer based on the provided model file name.
    It identifies the base model architecture (DistilBERT, MiniLM, MobileBERT) and loads the corresponding 
    model and tokenizer. The model is then moved to the specified device (e.g., GPU or CPU) and set to 
    evaluation mode.

    Args:
        model_file (str): The name of the file containing the saved model weights (e.g., 'distilbert_model.pth').
        model_load_path (str): The directory path where the model file is stored.
        device (str): The device to which the model should be moved (e.g., 'cuda' for GPU or 'cpu' for CPU).

    Returns:
        tuple: A tuple containing:
            - model_file (str): The name of the loaded model file.
            - tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
            - model (PreTrainedModel): The pre-trained model loaded with the state dict from the file.

    Raises:
        FileNotFoundError: If the model file does not exist in the specified path.
        OSError: If there is an error while loading the tokenizer or model (e.g., missing dependencies).
        RuntimeError: If there is a mismatch between the model architecture and the saved state dict.
    
    Notes:
        - The `num_labels` parameter should be defined globally in the scope or passed to the function.
        - The model is set to evaluation mode after loading, disabling dropout layers and enabling inference optimizations.
        - This function supports specific architectures: DistilBERT, MiniLM, and MobileBERT. If the model file 
          does not start with one of these, the function will not work as expected.
    """
    print(f"\nTrying to load model: {model_file}")
    
    try:
        # Identify the base model architecture
        if model_file.startswith("distilbert"):
            base_model = "distilbert-base-uncased"
        elif model_file.startswith("MiniLM"):
            base_model = "microsoft/MiniLM-L12-H384-uncased"
        elif model_file.startswith("mobilebert"):
            base_model = "google/mobilebert-uncased"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
        model_path = os.path.join(model_load_path, model_file)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        
        print(f"✅ Successfully loaded: {model_file}")

        return model_file, tokenizer, model

    except FileNotFoundError:
        print(f"❌ File not found: {model_file}")
    except OSError as e:
        print(f"❌ OSError while loading tokenizer/model: {model_file} | Error: {str(e)}")
    except RuntimeError as e:
        print(f"❌ RuntimeError (state dict or architecture mismatch): {model_file} | Error: {str(e)}")

def extract_probabilities(
        model_files: list[str]=None,
        model_load_path: str=None,
        device: str=None,
        train_texts: list[str]=None,
        train_labels: np.array=None,
        val_texts: list[str]=None,
        val_labels: np.array=None,
        test_texts: list[str]=None,
        test_labels: np.array=None
        ) -> dict:
    """
    Extracts the probabilities for each model in the given list of model files and computes their GM (Geometric Mean) score.

    This function iterates through a list of model files, loads each model, and makes predictions on a validation dataset.
    It computes the confusion matrix and the Geometric Mean (GM) score based on the recalls per class. If the GM score 
    exceeds a threshold of 0.70, the model's probabilities are stored in a dictionary along with the GM score.

    Args:
        model_files (list): A list of model file names (e.g., ['distilbert_model.pt', 'MiniLM_model.pt']).

    Returns:
        dict: A dictionary containing the models' names and their associated probabilities and GM score. 
              Example structure:
              {
                  'distilbert_model': {
                      'GM': 0.75,
                      'Probabilities': [0.1, 0.9, ...]
                  },
                  'MiniLM_model': {
                      'GM': 0.80,
                      'Probabilities': [0.2, 0.8, ...]
                  }
              }

    Notes:
        - Assumes `model_loader` and `data_loaders` functions are available and correctly defined in the scope.
        - Assumes the model files are saved with a '.pt' extension.
        - The function filters out models with GM scores less than or equal to 0.70.
        - Probabilities are returned as lists (as they are converted from numpy arrays).
        - The confusion matrix (`cm`) is calculated from the predicted labels and true labels (ground truth).
    """
    
    # Dictionary to store model names and their probabilities
    models_probs_dict = {}

    # Iterate over saved models in the folder
    for model_file in model_files:
        if model_file.endswith('.pt'):

            model_name, tokenizer, model = model_loader(model_file, model_load_path, device)
            _, val_loader, _ = data_loaders(
                tokenizer,
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                test_texts,
                test_labels
                )

            PREDS, PROBS, LABELS = predictor(model, val_loader, device)

            # Compute the confusion matrix and GM score
            cm = confusion_matrix(LABELS, PREDS)
            recalls_per_class = np.diag(cm) / np.sum(cm, axis=1)
            gm = np.prod(recalls_per_class) ** (1.0 / len(recalls_per_class))
            
            if gm > 0.70:
                models_probs_dict[model_name] = {
                    'GM': round(gm, 4),
                    'Probabilities': PROBS.tolist()  # Convert probabilities array to list for storage
                }

    return models_probs_dict

def cluster_creation(probs_list: np.array=None) -> np.array:
    """
    Creates clusters from a list of probabilities using Gaussian Mixture Model (GMM) after dimensionality reduction with UMAP.

    This function takes a list of model prediction probabilities, scales the data, reduces its dimensionality using UMAP, 
    and then applies e Gaussian Mixture Model (GMM) to find the best clustering solution based on the silhouette score. 
    The optimal number of clusters is determined by maximizing the silhouette score.

    Args:
        probs_list (list of np.ndarray): A list of model prediction probabilities (one for each model) for all samples.
                                         Each element in the list is an array of shape (num_samples, num_classes).

    Returns:
        np.ndarray: The cluster labels assigned to each sample based on the optimal number of clusters.
    
    Notes:
        - This function uses `StandardScaler` to normalize the data and `UMAP` for dimensionality reduction.
        - The optimal number of clusters is determined by maximizing the silhouette score.
        - `GaussianMixture` is used for clustering, and the best clustering is selected based on the silhouette score.
        - The silhouette score measures how similar each point is to its own cluster compared to other clusters.
        - Assumes the `umap` and `GaussianMixture` modules are imported and available in the environment.
    """
    
    # Stack the probabilities into a single array for clustering
    X = np.vstack(probs_list)

    # Standardize the data
    X = StandardScaler().fit_transform(X)

    # Set the reduced dimensionality to be two less than the number of samples
    red_dim = X.shape[0] - 2
    umap_reducer = umap.UMAP(n_components=red_dim, random_state=42)
    X_red = umap_reducer.fit_transform(X)

    # Initialize variables for tracking the best clustering solution
    best_num_clusters = 0
    best_silhouette = -1

    # Iterate over possible number of clusters (excluding the first two)
    _n = [_ for _ in range(X.shape[0])][2:]
    for num_clusters in _n:
        # Apply Gaussian Mixture Model (GMM) clustering
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(X_red)
        
        # Compute the silhouette score
        silhouette_avg = silhouette_score(X_red, cluster_labels)
        
        # Track the best clustering solution based on silhouette score
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_num_clusters = num_clusters
            best_cluster_labels = cluster_labels

    print('Best number of clusters:', best_num_clusters)
    cluster_labels = best_cluster_labels
    return cluster_labels


def eXperts_extraction(models_probs_dict: dict=None) -> np.array:
    """
    Extracts representative models from each cluster based on the proximity to the cluster's centroid.

    This function identifies clusters of models based on their prediction probabilities and selects the 
    most representative model in each cluster. The representative model is the one closest to the 
    centroid of its cluster in terms of Euclidean distance.

    Args:
        models_probs_dict (dict): A dictionary where keys are model names and values are dictionaries 
                                  containing 'Probabilities' (model prediction probabilities) and 
                                  'Cluster' (the cluster label assigned to the model). 
                                  Example:
                                  {
                                      'distilbert_model': {'Probabilities': [0.1, 0.9], 'Cluster': 0},
                                      'MiniLM_model': {'Probabilities': [0.2, 0.8], 'Cluster': 1}
                                  }

    Returns:
        dict: A dictionary containing the most representative model for each cluster. 
              The key is the model name, and the value is the dictionary with 'Probabilities' and 'Cluster'.
              Example:
              {
                  'distilbert_model': {'Probabilities': [0.1, 0.9], 'Cluster': 0},
                  'MiniLM_model': {'Probabilities': [0.2, 0.8], 'Cluster': 1}
              }

    Notes:
        - The function assumes that each model in `models_probs_dict` has a 'Cluster' label.
        - The Euclidean distance between the model's probability vector and the cluster centroid is used to identify the closest model.
        - The centroid of a cluster is computed as the mean of the probability vectors of the models in that cluster.
        - The resulting `squeezed_population` contains only one model per cluster, which is the most representative based on proximity to the centroid.
    """
    
    squeezed_population = {}

    # Step 1: Gather cluster labels and probability vectors
    model_names = list(models_probs_dict.keys())
    probs_matrix = np.vstack([models_probs_dict[name]['Probabilities'] for name in model_names])
    cluster_labels = [models_probs_dict[name]['Cluster'] for name in model_names]

    for cluster_id in set(cluster_labels):
        # Get all models in this cluster
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_probs = probs_matrix[indices]
        cluster_models = [model_names[i] for i in indices]

        # Compute the cluster centroid (mean of probabilities)
        centroid = np.mean(cluster_probs, axis=0)

        # Find the model closest to the centroid (based on Euclidean distance)
        distances = [np.linalg.norm(probs_matrix[i] - centroid) for i in indices]
        best_index = indices[np.argmin(distances)]
        best_model = model_names[best_index]

        # Add the representative model of this cluster to the squeezed population
        squeezed_population[best_model] = models_probs_dict[best_model]

    return squeezed_population

