import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.utils import SummaryFasta, kmersFasta
import argparse

def save_results(result, dataset, result_folder, run):
    file_path = os.path.join(result_folder, f'Supervised_Results_{dataset}.json')
    existing_data = {}
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    existing_data[run] = result
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=2)

def load_json_results(path, continue_flag):
    if os.path.isfile(path) and continue_flag:
        with open(path, 'r') as file:
            data = json.load(file)
        return {int(k): v for k, v in data.items()}
    return {}

def preprocess_data(args, summary_file):
    summary_dataset = pd.read_csv(summary_file, sep='\t', usecols=["Domain", "Temperature", "Assembly", "pH", "Genus", "Species"])
    fasta_file = os.path.join(args["results_folder"], args["Env"], f'Extremophiles_{args["Env"]}.fas')
    names, _, _, _ = SummaryFasta(fasta_file)
    assembly_dict = dict(zip(summary_dataset['Assembly'], summary_dataset.index))

    processed_data = {
        key: summary_dataset.loc[assembly_dict.get(name), key] if name in assembly_dict else None for name in names for key in ["Genus", "Species", "Domain", args["Env"]]
    }
    processed_data.update({"sequence_id": names, "Assembly": [assembly_dict.get(name) for name in names]})
    return pd.DataFrame(processed_data)

def supervised_classification(fasta_file, max_k, results, result_folder, env, exp):
    names, _, _, _ = SummaryFasta(fasta_file)
    results_dict = {k: {} for k in range(1, max_k + 1)}
    for k in range(1, max_k + 1):
        if k not in results:
            results_dict[k] = {}
        _, kmers = kmersFasta(fasta_file, k=k, reduce=True)
        kmers_normalized = np.transpose((np.transpose(kmers) / np.linalg.norm(kmers, axis=1)))
        perform_classification(kmers_normalized, results_dict, k, result_folder, env)
        print(f"Finished processing k = {k}")

    save_results(results, env, result_folder, exp)

def perform_classification(kmers, results, k, result_folder, env):
    classifiers = {
        "SVM": (SVC, {'kernel': 'rbf', 'class_weight': 'balanced', 'C': 10}),
        "Random Forest": (RandomForestClassifier, {}),
        "ANN": (MLPClassifier, {'hidden_layer_sizes': (256, 64), 'solver': 'adam',
                                'activation': 'relu', 'alpha': 1, 'learning_rate_init': 0.001,
                                'max_iter': 200, 'n_iter_no_change': 10})
    }


    env_file = os.path.join(result_folder, env, f'Extremophiles_{env}_GT_Env.tsv')
    tax_file = os.path.join(result_folder, env, f'Extremophiles_{env}_GT_Tax.tsv')
    for index, data_file in enumerate([env_file, tax_file]):
        for name, (algorithm, params) in classifiers.items():
            if name in results[k]:
                print(f"{name} classifier already computed for k = {k}-mers")
                continue
            results[k][name][index] = cross_validate_model(kmers, data_file, algorithm, params)

def cross_validate_model(kmers, label_file, algorithm, params):

    dataset = pd.read_csv(label_file, sep='\t')
    unique_labels = dataset['cluster_id'].unique().tolist()
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    idx = dataset.drop_duplicates(subset=['Assembly']).Assembly.to_numpy()
    y = dataset.drop_duplicates(subset=['Assembly']).cluster_id.to_numpy()
    scores = []
    for train_idx, test_idx in skf.split(idx, y):
        model = algorithm(**params)
        train_data = dataset.loc[dataset['Assembly'].isin(idx[train_idx])]
        test_data = dataset.loc[dataset['Assembly'].isin(idx[test_idx])]
        x_train = train_data[kmers.columns].to_numpy()
        y_train = [unique_labels.index(label) for label in train_data['cluster_id']]
        x_test = test_data[kmers.columns].to_numpy()
        y_test = [unique_labels.index(label) for label in test_data['cluster_id']]
        model.fit(x_train, y_train)
        scores.append(model.score(x_test, y_test))
    average_score = sum(scores) / len(scores)
    return average_score

def run(args):
    results_path = f'Supervised Results {args["Env"]}.json'
    results = load_json_results(results_path, args['continue'])
    fasta_file = os.path.join(args["results_folder"], args["Env"], 'Extremophiles_{args["Env"]}.fas')
    supervised_classification(fasta_file, args["max_k"], results, args["results_folder"], args["Env"], args["exp"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', action='store', type=str)
    parser.add_argument('--Env', action='store', type=str)  # [Temperature, pH]
    parser.add_argument('--n_clusters', action='store', type=int, default=None)  # [int]
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--exp', action='store', type=str)
    parser.add_argument('--max_k', action='store', type=int)

    args = vars(parser.parse_args())

    run(args)


if __name__ == '__main__':
    main()


