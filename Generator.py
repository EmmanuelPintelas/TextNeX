
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, confusion_matrix
from copy import deepcopy
device = torch.device("cuda")


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






# ---------------- Evaluation functions ------------------------------->
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
def Scores(predictions, probabilities, labels_query):

    # Accuracy: binary or multiclass classification
    acc = accuracy_score(labels_query, predictions)
    # F1 Score: Weighted average of the precision and recall
    f1 = f1_score(labels_query, predictions, average='macro')
    # Precision: Proportion of positive identifications that were actually correct
    pre = precision_score(labels_query, predictions, average='weighted', zero_division=0)
    # Recall: Proportion of actual positives that were correctly identified
    ##rec = recall_score(labels_query, predictions, average='weighted')

    # AUC: Area Under the ROC Curve, binary or multilabel classification
    probabilities = np.array(probabilities)
    if len(set(labels_query)) > 2:  # Multiclass case
        auc = roc_auc_score(labels_query, probabilities, multi_class='ovr', average='weighted')
    else:  # Binary case
        auc = roc_auc_score(labels_query, probabilities, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(labels_query, predictions)

    # GM: Geometric Mean metric, generalize for multiclass
    # We extract the diagonal of the confusion matrix and divide by the sum of each row
    recalls_per_class = np.diag(cm) / np.sum(cm, axis=1)
    # Calculate the geometric mean of the recall values
    gm = np.prod(recalls_per_class) ** (1.0 / len(recalls_per_class))

    return round(acc, 4), round(f1, 4), round(pre, 4), round(auc, 4), round(gm, 4), np.array(cm)

def run_evaluation(predictions, probabilities, labels_query):
    """Evaluates predictions and appends metrics to evaluation_metrics."""
    acc, f1, pre, auc, gm,   cm = Scores(predictions, probabilities, labels_query)
    ##print('----> ', acc, f1, pre, auc, gm, ' <----\n')
    evaluation_metrics['Acc'].append(acc)
    evaluation_metrics['F1'].append(f1)
    evaluation_metrics['Pre'].append(pre)
    evaluation_metrics['AUC'].append(auc)
    evaluation_metrics['GM'].append(gm)
    evaluation_metrics['cm'].append(cm)
# <---------------- Evaluation functions ---------------------------------


# ---------------- Training functions ----------------------------------->
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min'):
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

    def check_early_stop(self, current_score):
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

# Basic Data Augmentation (to think to add: reverse the word, synonym replacement)
def augment_text(text):
    words = text.split()
    num_replacements = random.randint(1, max(1, len(words) // 10))
    for _ in range(num_replacements):
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx][::-1]  # <-- reverse the word
    return " ".join(words)
# <---------------- Training functions -------------------------------------------



# ----------   Hyperparameter sets  ------------------------>
N = 50 # the number of trained models to generate
learning_rates_inits = [3e-5, 5e-5, 1e-5, 5e-6]
weight_decays = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
batch_sizes = [32, 64]
num_epochs = [15, 20, 30, 40, 50]
schedul_patiences = [2, 3, 4, 5]
schedul_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
early_stop_patiences = [5, 6, 7, 8]
max_length = 106
# <--------------------------------------------------------- 


# -------- Storage folders------------>
root_path_save = 'Storage'
folders = ["models", "metrics"]

paths_save = [os.path.join(root_path_save, folder) for folder in folders]
[os.makedirs(path_save, exist_ok=True) for path_save  in paths_save]
# <-----------------------------------


# -------------------------------------------- Generation Phase Starts --------------------------------------------------------------->
for it in range(N):
    evaluation_metrics = {'Acc': [], 'F1': [], 'Pre': [], 'AUC': [] , 'GM': [], 'cm':[]}

    print(f"Generating Model {it+1}/{N}")

    # --- Randomly Initialize new TextNet ----------->
    model_load_path = "distilbert-base-uncased"
    # model_name = "distilbert-base-uncased"# "microsoft/MiniLM-L12-H384-uncased"# "google/mobilebert-uncased"

    model_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_load_path, num_labels=num_labels).to(device)

    # Random sample hyperparameters ------------>
    lr_init = random.choice(learning_rates_inits)
    wei_decay = random.choice(weight_decays)
    batch_size = random.choice(batch_sizes)
    epochs = random.choice(num_epochs)
    schedul_patience = random.choice(schedul_patiences) 
    schedul_factor = random.choice(schedul_factors)
    early_stop_patience = random.choice(early_stop_patiences)
     

    print(f'\n----- Iter: {it} -----')
    #print(f'----- epochs: {epochs} - lr_init: {lr_init} - bs: {batch_size} - max_length: {max_length} ---\n')
    print(f'----- epochs: {epochs} - schedul_patience: {schedul_patience} - lr_init: {lr_init} - bs: {batch_size}---\n')


    # ---------- Freeze first two layers --------------------->
    # for name, param in model.named_parameters():
    #     if "encoder.layer.0" in name or "encoder.layer.1" in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True


    optimizer = optim.Adam(model.parameters(), lr = lr_init, weight_decay = wei_decay)# optim.Adam # or # AdamW
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience = schedul_patience, factor = schedul_factor)
    early_stopping = EarlyStopping(patience=early_stop_patience, mode='max')


    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length = max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length = max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length = max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # -------------------------------- Training loop -------------------------------------------------------->
    best_f1 = 0
    useless = 0
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        train_preds, train_labels_batch = [], []
        for input_ids, attention_mask, label in train_loader:
            input_ids, attention_mask, label = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    label.to(device),
                )
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels_batch.extend(label.cpu().numpy())
        train_accuracy = accuracy_score(train_labels_batch, train_preds)
        train_f1 = f1_score(train_labels_batch, train_preds, average="macro")


        # ----------- Validation and Storaging ------------------------------------------>
        model.eval()
        val_preds, val_probs, val_labels = [], [], []
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, label in val_loader:
                input_ids, attention_mask, label = (
                        input_ids.to(device),
                        attention_mask.to(device),
                        label.to(device),
                    )
                outputs = model(input_ids, attention_mask=attention_mask, labels=label)
                val_loss += outputs.loss.item()
                logits = outputs.logits
                preds = logits.argmax(dim=1).cpu().numpy()
                probs = logits.softmax(dim=1)[:, 1].cpu().numpy()
                val_preds.extend(preds)
                val_probs.extend(probs)
                val_labels.extend(label.cpu().numpy())
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        scheduler.step(val_f1)

        # --------- Storaging (models and full metrics) ------------------>
        if val_f1 > best_f1:
            best_f1 = val_f1
            print('----> ', np.round(best_f1,5), ' <----\n')
            if val_f1 > 0.60:

                model_save   = model_name + '_' + str(np.round(val_f1,5)) + '.pt'
                metrics_save = model_name + '_' + str(np.round(val_f1,5)) + '.xlsx'

                torch.save(model.state_dict(), os.path.join(paths_save[0], model_save))

                run_evaluation(val_preds, val_probs, val_labels)
                metrics_df = pd.DataFrame(evaluation_metrics)

                metrics_df.to_excel(os.path.join(paths_save[1], metrics_save), index=False)
        # < -------- Storaging (models and full metrics) -----------------


        if val_f1 > 0.50:
            early_stopping.check_early_stop(val_f1)
        else:
            useless += 1
            if useless == 10:
                print("Early stopping - Useless")
                break
        if early_stopping.early_stop:
            print("Early stopping - Overfit")
            break

        print(f"Epoch {epoch+1}/{epochs}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")



