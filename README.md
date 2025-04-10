# TextNeX

Heterogeneous ensemble of lightweight Text Network eXperts.

TextNet-HeX is a modular and efficient framework for text classification using an ensemble of lightweight transformer-based models. It consists of three main phases: Generator, Selector, and Aggregator.

## 🔧 Phase 1: Generator

This phase generates multiple lightweight text networks, each trained under diverse hyperparameter configurations to capture unique perspectives on text characteristics.

Steps:

Step 1: Specify root path of data, e.g.  
root = 'Datasets'

Step 2: Specify Task, e.g.  
task = 'substask_1' # corrsponds to AuTexTification dataset

Step 3: Specify number of iterations, e.g.  
N = 50  # the number of trained models to generate

Step 4: Specify save path, e.g.  
root_path_save = 'Storage'

Step 5: Specify validation F1 threshold to save only useful models, e.g.  
val_f1 > 0.62

Step 6: Specify the model architecture to load, e.g.  
model_load_path = "distilbert-base-uncased"
  # Alternatives:
  # model_name = "microsoft/MiniLM-L12-H384-uncased"
  # model_name = "google/mobilebert-uncased"

Step 7: Run the script. After training completes, your output will be:

Storage/  
├── models/    → Saved models (e.g., distilbert_0.7435.pt)  
└── metrics/   → Excel files containing evaluation metrics of each saved model

## 🔍 Phase 2: Selector

This phase filters and selects a small, diverse set of expert models based on their prediction behavior.

What it does:

- Loads all models from `Storage/models/`
- Filters models with Geometric Mean (GM) > 0.70
- Extracts validation prediction probabilities
- Applies UMAP for dimensionality reduction
- Applies Gaussian Mixture Model (GMM) clustering
- Selects the most representative model (closest to cluster centroid) per cluster
- Copies the selected experts to `Storage/HeX/`

Output:

Storage/  
└── HeX/   → Folder containing the selected expert models

## 🤝 Phase 3: Aggregator

This phase uses Sequential Quadratic Programming (SQP) to compute the optimal weights for aggregating the selected experts.

What it does:

- Loads all models from `Storage/HeX/`
- Extracts logits on the validation set
- Learns optimal weights to maximize Geometric Mean (GM) across validation predictions
- Applies the weighted ensemble on the test set for final evaluation

Example output:

Optimal Weights: [0.32, 0.18, 0.50]  
Best GM Achieved: 0.8451

📊 Final Test Performance  
Accuracy: 0.775 | AUC: 0.848 | GM: 0.772

## 🚀 End-to-End Usage

```bash
# Step 1: Generate models
python Generator.py

# Step 2: Select expert models
python Selector.py

# Step 3: Aggregate and evaluate
python Aggregator.py
