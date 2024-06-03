import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from src.utils import kmersFasta  # Assuming you have this utility function

# Function to train a Random Forest and compute MDI feature importance
def mdi_feature_importance(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    return importances

# Function to compute Permutation Feature Importance
def permutation_feature_importance(model, X_test, y_test):
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring='accuracy')
    return perm_importance.importances_mean

# Function to compute Mean Decrease Accuracy (MDA)
def mda_feature_importance(model, X_train, y_train, X_test, y_test):
    baseline_score = accuracy_score(y_test, model.predict(X_test))
    mda_scores = []
    for i in range(X_train.shape[1]):
        X_train_permuted = X_train.copy()
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_train_permuted[:, i])
        np.random.shuffle(X_test_permuted[:, i])
        permuted_score = accuracy_score(y_test, model.predict(X_test_permuted))
        mda_scores.append(baseline_score - permuted_score)
    return np.array(mda_scores)

def read_fasta(file_path):

    id_2_sequences = {}
    # Open the FASTA file
    with open(file_path, 'r') as file:
        # Iterate over each record
        for record in SeqIO.parse(file, 'fasta'):
            id_2_sequences[str(record.id)] = str(record.seq)

    return id_2_sequences

env = 'Temperature'
env = 'pH'
tax = 'Domain'

# Load dataset and extract k-mers (replace with your actual dataset)
data_path = ''
label_path = ''
fasta_file = read_fasta(file_path)
labels = pd.read_csv(label_path)['']

_, kmers = kmersFasta(fasta_file, k=6, transform=None, reduce=True)
kmers_normalized = np.transpose((np.transpose(kmers) / np.linalg.norm(kmers, axis=1)))
print(kmers_normalized)

X = kmers
y =

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
svm = SVC(kernel='rbf', class_weight= 'balanced', C= 10)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
ann = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, n_iter_no_change=10, learning_rate_init = 0.001, alpha = 1, activation = 'relu', solver = 'adam')

svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
ann.fit(X_train, y_train)

# Compute feature importances
mdi_importances = mdi_feature_importance(X_train, y_train)
perm_importances_svm = permutation_feature_importance(svm, X_test, y_test)
perm_importances_rf = permutation_feature_importance(rf, X_test, y_test)
perm_importances_ann = permutation_feature_importance(ann, X_test, y_test)
mda_importances_svm = mda_feature_importance(svm, X_train, y_train, X_test, y_test)
mda_importances_rf = mda_feature_importance(rf, X_train, y_train, X_test, y_test)
mda_importances_ann = mda_feature_importance(ann, X_train, y_train, X_test, y_test)

# Aggregate results
feature_importances = {
    'mdi': mdi_importances,
    'perm_svm': perm_importances_svm,
    'perm_rf': perm_importances_rf,
    'perm_ann': perm_importances_ann,
    'mda_svm': mda_importances_svm,
    'mda_rf': mda_importances_rf,
    'mda_ann': mda_importances_ann
}

# Convert to DataFrame for easier handling and visualization
feature_importances_df = pd.DataFrame(feature_importances)

# Plot the feature importances
feature_importances_df.plot(kind='bar', figsize=(15, 8))
plt.title('Feature Importances from Different Methods')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Save the aggregated results to a CSV file
feature_importances_df.to_csv('feature_importances.csv', index=False)
