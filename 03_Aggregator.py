import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.aggegator_utils import (
    extract_model_outputs,
    SQP,
    test_ensemble
    )

if __name__ == "__main__":
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

    logits_list = extract_model_outputs(
        models=models,
        tokenizers=tokenizers,
        device=device
        )
    # Run SQP
    optimal_weights, best_gm = SQP(logits_list, np.array(val_labels))
    print("\nOptimal Weights:", optimal_weights)
    print("Best GM Achieved:", best_gm)
    # Run test ensemble with learned weights
    acc, f1, pre, auc, gm, cm = test_ensemble(
        models=models,
        weights=optimal_weights,
        tokenizer=tokenizers[0],
        device=device
        )
    print("\n📊 Final Test Performance")
    print(f"Accuracy: {acc} | F1: {f1} | Precision: {pre} | AUC: {auc} | GM: {gm}")
    print("Confusion Matrix:\n", cm)


