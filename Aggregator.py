import os
import json
import random
import numpy as np
import pandas as pd
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from scipy.optimize import minimize
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load data ----------------------------------->
root = 'Datasets'
task = 'substask_1'
files = ['train_split.csv', 'val_split.csv', 'test_split.csv']

data_paths = [os.path.join(root, task, file) for file in files]
data_df = [pd.read_csv(path) for path in data_paths]

train_texts, val_texts, test_texts = [data['text'].tolist() for data in data_df]
train_labels, val_labels, test_labels = [data['label'].tolist() for data in data_df]

label_mapping = {label: idx for idx, label in enumerate(set(train_labels))}
train_labels = [label_mapping[label] for label in train_labels]
val_labels = [label_mapping[label] for label in val_labels]
test_labels = [label_mapping[label] for label in test_labels]

num_labels = len(label_mapping)

# Dataset class
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

def get_val_loader(tokenizer):
    max_length = 106
    batch_size = 256
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return val_loader

def extract_model_outputs(models, tokenizers):
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

def Scores(predictions, probabilities, labels_query):
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

evaluation_metrics = {'Acc': [], 'F1': [], 'Pre': [], 'AUC': [], 'GM': [], 'cm': []}

def run_evaluation(predictions, probabilities, labels_query):
    acc, f1, pre, auc, gm, cm = Scores(predictions, probabilities, labels_query)
    evaluation_metrics['Acc'].append(acc)
    evaluation_metrics['F1'].append(f1)
    evaluation_metrics['Pre'].append(pre)
    evaluation_metrics['AUC'].append(auc)
    evaluation_metrics['GM'].append(gm)
    evaluation_metrics['cm'].append(cm)

best_gm = -float('inf')
best_weights = None

def ensemble_loss(weights, logits_list, labels):
    global best_gm, best_weights

    weights = np.array(weights)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    stacked_logits = torch.stack(logits_list)  # [n_models, batch, n_classes]
    weighted_logits = torch.sum(weights_tensor * stacked_logits, dim=0)

    probs = torch.softmax(weighted_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    prob_positives = probs[:, 1] if probs.shape[1] == 2 else probs
    _, _, _, _, gm, _ = Scores(preds, prob_positives, labels)

    if gm > best_gm:
        print(f"\n✅ New Best GM: {gm:.4f}")
        best_gm = gm
        best_weights = weights.copy()

    return -gm

def SQP(logits_list, labels):
    global best_gm, best_weights
    best_gm = -float('inf')
    best_weights = None

    num_models = len(logits_list)
    initial_weights = np.ones(num_models) / num_models
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(num_models)]

    result = minimize(ensemble_loss, initial_weights, args=(logits_list, labels), method='Powell', bounds=bounds, constraints=constraints)

    if best_weights is not None:
        return best_weights
    else:
        raise ValueError("Optimization failed to improve GM")

# Load models
model_load_path = "Storage/HeX"
model_files = os.listdir(model_load_path)

model_names, tokenizers, models = [], [], []
for model_file in model_files:
    if model_file.endswith(".pt"):
        if model_file.startswith("distilbert"):
            base_model = "distilbert-base-uncased"
        elif model_file.startswith("MiniLM"):
            base_model = "microsoft/MiniLM-L12-H384-uncased"
        elif model_file.startswith("mobilebert"):
            base_model = "google/mobilebert-uncased"
        else:
            continue

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
        model_path = os.path.join(model_load_path, model_file)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        model_names.append(model_file)
        tokenizers.append(tokenizer)
        models.append(model)


logits_list = extract_model_outputs(models, tokenizers)

# Run SQP
optimal_weights = SQP(logits_list, np.array(val_labels))
print("\nOptimal Weights:", optimal_weights)
print("Best GM Achieved:", best_gm)





def get_test_loader(tokenizer):
    max_length = 106
    batch_size = 256
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader

def test_ensemble(models, weights, tokenizer):
    test_loader = get_test_loader(tokenizer)
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

    acc, f1, pre, auc, gm, cm = Scores(np.array(all_preds), np.array(all_probs), np.array(all_labels))
    print("\n📊 Final Test Performance")
    print(f"Accuracy: {acc} | F1: {f1} | Precision: {pre} | AUC: {auc} | GM: {gm}")
    print("Confusion Matrix:\n", cm)

# Run test ensemble with learned weights
test_ensemble(models, optimal_weights, tokenizers[0])


