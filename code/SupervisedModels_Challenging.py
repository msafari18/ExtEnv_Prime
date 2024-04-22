import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.utils import SummaryFasta, kmersFasta
import torch

def save_results(results, dataset, result_folder, run):
    file_path = os.path.join(result_folder, f'Supervised_Results_no_Genus_new{dataset}.json')
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = {}
    existing_data[run] = results
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=2)

def load_results(file_path, continue_flag):
    if os.path.isfile(file_path) and continue_flag:
        with open(file_path, 'r') as file:
            return {int(k): v for k, v in json.load(file).items()}
    return {}

def preprocess_data(fasta_file, summary_file):
    names, _, _, _ = SummaryFasta(fasta_file)
    summary_dataset = pd.read_csv(summary_file, sep='\t',
                                  usecols=["Domain", "Temperature", "Assembly", "pH", "Genus", "Species"])
    assembly_dict = {name: summary_dataset[summary_dataset["Assembly"] == name].iloc[0]
                     for name in names if summary_dataset["Assembly"].eq(name).any()}
    data = pd.DataFrame({
        "Domain": [info["Domain"] for info in assembly_dict.values()],
        "Genus": [info["Genus"] for info in assembly_dict.values()],
        "Species": [info["Species"] for info in assembly_dict.values()],
        "Assembly": [info["Assembly"] for info in assembly_dict.values()],
        "sequence_id": list(assembly_dict.keys())
    })
    return data

def perform_classification(fasta_file, results, result_folder, env, max_k, exp):
    data = preprocess_data(fasta_file, "/path/to/Extremophiles_GTDB.tsv")
    classifiers = {
        "SVM": (SVC, {'kernel': 'rbf', 'class_weight': 'balanced', 'C': 10}),
        "Random Forest": (RandomForestClassifier, {}),
        "ANN": (MLPClassifier, {'hidden_layer_sizes': (256, 64), 'solver': 'adam',
                                'activation': 'relu', 'alpha': 1, 'learning_rate_init': 0.001,
                                'max_iter': 200, 'n_iter_no_change': 10})
    }
    env_file = os.path.join(result_folder, env, f'Extremophiles_{env}_GT_Env.tsv')
    tax_file = os.path.join(result_folder, env, f'Extremophiles_{env}_GT_Tax.tsv')

    for k in range(1, max_k + 1):
        _, kmers = kmersFasta(fasta_file, k=k, reduce=True)
        for index, label_file in enumerate([env_file, tax_file]):
            results.setdefault(k, {})
            for name, (algorithm, params) in classifiers.items():
                if name in results[k]:
                    continue
                results[k][name][index] = cross_validate_model(data, kmers, label_file, algorithm, params)

    save_results(results, env, result_folder, exp)

def cross_validate_model(data, kmers, label_file, algorithm, params):
    dataset = pd.read_csv(label_file, sep='\t')
    unique_labels = dataset['cluster_id'].unique()
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True)
    scores = []
    for train_idx, test_idx in skf.split(dataset, dataset['cluster_id'], groups=data["genus"].values):
        model = algorithm(**params)
        train_dataset = dataset.iloc[train_idx]
        test_dataset = dataset.iloc[test_idx]
        x_train = kmers[train_dataset.index]
        y_train = [np.where(unique_labels == label)[0][0] for label in train_dataset['cluster_id']]
        x_test = kmers[test_dataset.index]
        y_test = [np.where(unique_labels == label)[0][0] for label in test_dataset['cluster_id']]
        model.fit(x_train, y_train)
        scores.append(model.score(x_test, y_test))
    return np.mean(scores)

def run(args):
    results_path = f'Supervised Results no Genus {args["Env"]}.json'
    results = load_results(results_path, args['continue'])
    fasta_file = os.path.join(args["results_folder"], args["Env"], f'Extremophiles_{args["Env"]}.fas')
    perform_classification(fasta_file, results, args["results_folder"], args["Env"], args["max_k"], args["exp"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', required=True, type=str)
    parser.add_argument('--Env', required=True, type=str)
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--exp', required=True, type=str)
    parser.add_argument('--max_k', required=True, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(vars(args))

if __name__ == '__main__':
    main()
