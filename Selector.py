import os
import json
import random
import numpy as np
import pandas as pd
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, confusion_matrix
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Load data ----------------------------------->
import os
import numpy as np
import pandas as pd

root = 'Datasets'

task = 'substask_1'# 'substask_2'
files = ['train_split.csv', 'val_split.csv', 'test_split.csv']

data_paths = [os.path.join(root,task,file) for file in files]
data_df = [pd.read_csv(path) for path in data_paths]

train_texts, val_texts, test_texts = [data['text'].tolist() for data in data_df]
train_labels, val_labels, test_labels = [data['label'].tolist() for data in data_df]


# Convert labels to integers
label_mapping = {label: idx for idx, label in enumerate(set(train_labels))}
train_labels = [label_mapping[label] for label in train_labels]
val_labels = [label_mapping[label] for label in val_labels]
test_labels = [label_mapping[label] for label in test_labels]

num_labels = len(label_mapping)
# <---------------- Load data ---------------------------------------


# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
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

def data_loaders (tokenizer):
    max_length = 106
    batch_size=256
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length = max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length = max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length = max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    return train_loader, val_loader, test_loader

def predictor(model, dataloader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
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





def model_loader (model_file):
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


def Extract_Probabilities(model_files):

    # Dictionary to store model names and their probabilities
    models_probs_dict = {}

    # Iterate over saved models in the folder
    for model_file in model_files:
        if model_file.endswith('.pt'):

            model_name, tokenizer, model = model_loader(model_file)
            _, val_loader, _ = data_loaders (tokenizer)

            PREDS, PROBS, LABELS = predictor(model, val_loader)

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

from sklearn.preprocessing import StandardScaler
import umap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
def cluster_creation(probs_list):

    X = np.vstack(probs_list)

    X = StandardScaler().fit_transform(X)

    red_dim = X.shape[0] - 2
    umap_reducer = umap_reducer = umap.UMAP(n_components=red_dim, random_state=42)
    X_red = umap_reducer.fit_transform(X)

    best_num_clusters = 0
    best_silhouette = -1

    _n = [_ for _ in range(X.shape[0])][2:]
    for num_clusters in _n:
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(X_red)
        silhouette_avg = silhouette_score(X_red, cluster_labels)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_num_clusters = num_clusters
            best_cluster_labels = cluster_labels
    print('best number clusters: ', best_num_clusters)
    cluster_labels = best_cluster_labels
    return cluster_labels

def eXperts_extraction(models_probs_dict):
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

        # Compute the cluster centroid
        centroid = np.mean(cluster_probs, axis=0)

        # Find the model closest to the centroid
        distances = [np.linalg.norm(probs_matrix[i] - centroid) for i in indices]
        best_index = indices[np.argmin(distances)]
        best_model = model_names[best_index]

        # Add the representative model of this cluster
        squeezed_population[best_model] = models_probs_dict[best_model]

    return squeezed_population



model_load_path = "Storage/models"# <-- the trained textnets based on generation phase
save_dir = "Storage/HeX" # <-- the selected textnet experts models to be considered in the aggregation phase of final Ensemble optimization phase 

model_files = os.listdir(model_load_path)
models_probs_dict = Extract_Probabilities(model_files)
# with open('models_probs_dict.json', 'w') as json_file:
#         json.dump(models_probs_dict, json_file, indent=4)
# with open('models_probs_dict.json', 'r') as json_file:
# models_probs_dict = json.load(json_file)

# scaling probs and update dictionary
model_names = list(models_probs_dict.keys())
probs_list = [np.array(models_probs_dict[model_name]['Probabilities']) for model_name in model_names]
X = np.vstack(probs_list)
scaler_init = StandardScaler()
X = scaler_init.fit_transform(X)
for model_name, x in zip(model_names, X):
    models_probs_dict[model_name]['Probabilities'] = x


# --------------- clustering  --------------- #
model_names = list(models_probs_dict.keys())
probs_list = [np.array(models_probs_dict[model_name]['Probabilities']) for model_name in model_names]
cluster_labels = cluster_creation(probs_list)
# Update models_probs with the clusters
for model_name, cluster_label in zip(model_names, cluster_labels):
    models_probs_dict[model_name]['Cluster'] = int(cluster_label)

# --------------- centroid-based eXperts extraction  --------------- #
selected_population = eXperts_extraction(models_probs_dict)
model_names = list(selected_population.keys())
for m in model_names:
    source_path = os.path.join(model_load_path, m)
    destination_path = os.path.join(save_dir, m)
    shutil.copy(source_path , destination_path)