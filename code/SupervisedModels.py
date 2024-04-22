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


def supervised_classification(fasta_file, max_k, results, result_folder, env, exp):
    for k in range(1, max_k + 1):
        _, kmers = kmersFasta(fasta_file, k=k, transform=None, reduce=True)
        kmers_normalized = np.transpose((np.transpose(kmers) / np.linalg.norm(kmers, axis=1)))
        results = {key: {} for key in range(1, max_k + 1)}
        perform_classification(kmers_normalized, k, results, result_folder, env, exp)
        print(f"Finished processing k = {k}")


def perform_classification(kmers, k, results, result_folder, env, exp):
    classifiers = {
        "SVM": (SVC, {'kernel': 'rbf', 'class_weight': 'balanced', 'C': 10}),
        "Random Forest": (RandomForestClassifier, {}),
        "ANN": (MLPClassifier, {'hidden_layer_sizes': (256, 64), 'solver': 'adam',
                                'activation': 'relu', 'alpha': 1, 'learning_rate_init': 0.001,
                                'max_iter': 300, 'n_iter_no_change': 10})
    }

    env_file = os.path.join(result_folder, env, f'Extremophiles_{env}_GT_Env.tsv')
    tax_file = os.path.join(result_folder, env, f'Extremophiles_{env}_GT_Tax.tsv')

    for name, (algorithm, params) in classifiers.items():
        results[k][name] = [0, 0]
        for index, label_file in enumerate([env_file, tax_file]):
            label_data = pd.read_csv(label_file, sep='\t')
            results[k][name][index] = cross_validate_model(kmers, label_data, algorithm, params)
    save_results(results, env, result_folder, exp)


def cross_validate_model(kmers, label_data, algorithm, params):
    unique_labels = list(label_data['cluster_id'].unique())
    Dataset = pd.concat([label_data, pd.DataFrame(kmers)], axis=1)
    Dataset.dropna(inplace=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    idx = Dataset.drop_duplicates(subset=["Assembly"]).Assembly.to_numpy()
    y = Dataset.drop_duplicates(subset=["Assembly"]).cluster_id.to_numpy()
    n_samples, n_features = kmers.shape
    print(len(Dataset), n_features)
    scores = []

    for train, test in skf.split(idx, y):
        model = algorithm(**params)
        train_Dataset = Dataset.set_index("Assembly").loc[idx[train]].reset_index()
        test_Dataset = Dataset.set_index("Assembly").loc[idx[test]].reset_index()

        x_train = train_Dataset.loc[:, list(range(n_features))].to_numpy()
        y_train = list(map(lambda x: unique_labels.index(x),
                           train_Dataset.loc[:, ['cluster_id']].to_numpy().reshape(-1)))

        x_test = test_Dataset.loc[:, list(range(n_features))].to_numpy()
        y_test = list(map(lambda x: unique_labels.index(x),
                          test_Dataset.loc[:, ['cluster_id']].to_numpy().reshape(-1)))

        model.fit(x_train, y_train)
        scores.append(model.score(x_test, y_test))

        del model

    average_score = sum(scores) / len(scores)
    return average_score


def run(args):
    results_path = f'Supervised_Results_{args["Env"]}.json'
    results = load_json_results(results_path, args['continue'])
    fasta_file = os.path.join(args["results_folder"], args["Env"], f'Extremophiles_{args["Env"]}.fas')
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


