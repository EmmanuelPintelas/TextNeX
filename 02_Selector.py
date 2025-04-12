import os
import torch
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.selector_utils import( 
    extract_probabilities,
    cluster_creation,
    eXperts_extraction
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model_load_path = "Storage/models"# <-- the trained textnets based on generation phase
    save_dir = "Storage/HeX" # <-- the selected textnet experts models to be considered in the aggregation phase of final Ensemble optimization phase 
    os.makedirs(save_dir, exist_ok=True)

    model_files = os.listdir(model_load_path)
    models_probs_dict = extract_probabilities(
        model_files,
        model_load_path,
        device,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels
        )
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
