import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.metrics import homogeneity_completeness_v_measure as cluster_quality
import os
import sys
from sklearn.metrics import completeness_score, homogeneity_score

sys.path.append('src')
import vamb
from models import CLE, VAE_encoder
from idelucs.cluster import iDeLUCS_cluster
from utils import kmersFasta

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})

# Configuration and data preparation

TAX_LEVEL = 'Genus'


# Dimensionality reduction functions
def VAE(sequence_file, latent_file=None):
    with vamb.vambtools.Reader(sequence_file) as filehandle:
        composition = vamb.parsecontigs.Composition.from_file(filehandle)

    rpkms = np.ones((composition.matrix.shape[0], 1), dtype=np.float32)
    vae = vamb.encode.VAE(nsamples=1, nlatent=32)
    dataloader = vamb.encode.make_dataloader(
        rpkms,
        composition.matrix,
        composition.metadata.lengths,
    )

    vae.trainmodel(dataloader)
    latent = vae.encode(dataloader)
    names = composition.metadata.identifiers

    return names, latent

def CL(sequence_file=None, latent_file=None):
    cl = CLE(sequence_file)
    names, latent = cl.encode()
    return names, latent


def UMAP(sequence_file=None, latent_file=None):
    names, kmers = kmersFasta(sequence_file, k=6)
    return names, umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.0, n_components=64).fit_transform(kmers)


def iDeLUCS(sequence_file=None, params=None):
    model = iDeLUCS_cluster(**params)
    return model.fit_predict(sequence_file)


# Clustering functions
def IM(latent, names):
    cluster_iterator = vamb.cluster.cluster(latent.astype(np.float32), labels=names)
    cluster_dict = {}
    for cluster_name, seq_names in cluster_iterator:
        for seq_id in seq_names:
            cluster_dict.setdefault(seq_id, []).append(cluster_name)
    return np.array([cluster_dict[_id][0] for _id in names if _id in cluster_dict])


def DBSCAN(latent, names):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True).fit(latent)
    return clusterer.labels_


def insert_assignment(summary_dataset, assignment_algo, names, labels):
    """
    Updates the summary dataset with new cluster assignments and calculates cluster quality metrics using the specified taxonomic level.

    Parameters:
        summary_dataset (DataFrame): The dataset to update.
        assignment_algo (str): The name of the algorithm used for the assignment.
        names (list): List of sequence names or identifiers.
        labels (list): List of labels corresponding to the names.
    """
    # Create a mapping from names to labels for easier updates.
    name_to_label = dict(zip(names, labels))
    # Update the summary dataset with new labels.
    summary_dataset[assignment_algo] = summary_dataset['Assembly'].map(name_to_label)

    # Calculate and print the clustering quality metrics.
    if TAX_LEVEL in summary_dataset:
        GT = summary_dataset[TAX_LEVEL].dropna().to_numpy()
        unique_labels = np.unique(GT)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_mapping[genus] for genus in GT])
        y_pred = summary_dataset[assignment_algo].dropna().map(label_mapping).to_numpy()
        results = cluster_quality(y, y_pred)
        print(f'Clustering Quality for {assignment_algo}:', results)
    else:
        print(f'Column "{TAX_LEVEL}" not found in summary_dataset for clustering quality calculation.')


def run_models(env, path, fragment_length, k):

    result_folder = f"{path}/exp2/0/fragments_{fragment_length}"

    summary_file = f'{path}/Extremophiles_GTDB.tsv'
    summary_dataset = pd.read_csv(summary_file, sep='\t',
                                  usecols=["Assembly", "Domain", env,
                                           "pH", "Phylum", "Class", "Order",
                                           "Family", "Genus", "Species"])
    summary_dataset.dropna(subset=[env], inplace=True)

    sequence_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')

    models = {
        "VAE": VAE,
        "UMAP": UMAP,
        "iDeLUCS": lambda sequence_file: iDeLUCS_cluster(sequence_file, n_clusters=200, n_epochs=50,
                                                         n_mimics=3, batch_sz=512, k=k, weight=0.25,
                                                         n_voters=5, optimizer='AdamW', scheduler='Linear')
    }

    clust_algorithms = {"IM": IM, "HDBSCAN": DBSCAN}

    for model_name, model_func in models.items():
        print(f"......... Processing {model_name} ...............")
        if model_name == "iDeLUCS":
            labels, latent = model_func(sequence_file)
            names = [label[0] for label in labels]  # Assuming labels includes names
            labels = [label[1] for label in labels]  # Assuming labels includes actual cluster labels
        else:
            names, latent = model_func(sequence_file)
            for clust_name, clust_func in clust_algorithms.items():
                labels = clust_func(latent, names)
                assignment_algo = f'{model_name}+{clust_name}'
                insert_assignment(summary_dataset, assignment_algo, names, labels)
        if model_name == "iDeLUCS":
            insert_assignment(summary_dataset, model_name, names, labels)


def analyze_result(summary_dataset, dataset_column, dim_algorithms, clust_algorithms):
    """
    Analyzes the dataset to count significant taxa for each environmental condition
    and generates a list of algorithm names based on dimensionality and clustering methods.

    Parameters:
        summary_dataset (pd.DataFrame): The DataFrame containing the dataset.
        dataset_column (str): The name of the column in the DataFrame to analyze.
        dim_algorithms (list): A list of tuples representing dimensionality reduction algorithms.
        clust_algorithms (list): A list of tuples representing clustering algorithms.
    """
    env_taxonomic_counts = {}

    for env_value in summary_dataset[dataset_column].unique():
        env_subset = summary_dataset[summary_dataset[dataset_column] == env_value]
        taxonomic_counts = env_subset[TAX_LEVEL].value_counts()
        significant_taxa = taxonomic_counts[taxonomic_counts >= 2]
        env_taxonomic_counts[env_value] = len(significant_taxa)

    df_env_taxa = pd.DataFrame(list(env_taxonomic_counts.items()), columns=[dataset_column, 'Significant Taxa Count'])

    algo_names = [f'{dim_name}+{clust_name}' for dim_name, _ in dim_algorithms for clust_name, _ in clust_algorithms]
    algo_names.append('iDeLUCS')  # Add the standalone iDeLUCS algorithm

    return df_env_taxa, algo_names


def analyze_clustering_results(summary_dataset, dataset_column, algo_names):
    """
    Process and analyze clustering results to calculate completeness and homogeneity metrics and visualize outcomes.
    """

    GOOD_CLUSTERS = {}
    overall_results = pd.DataFrame()  # To store summary results for visualization

    for name in algo_names:
        results_df = pd.DataFrame()
        results_df['predicted_cluster'] = summary_dataset[name]
        results_df['true_genus'] = summary_dataset['Genus']

        # Calculate completeness and homogeneity using built-in sklearn functions
        completeness = completeness_score(results_df['true_genus'], results_df['predicted_cluster'])
        homogeneity = homogeneity_score(results_df['true_genus'], results_df['predicted_cluster'])

        # Prepare data for visualization and further analysis
        cluster_info = summary_dataset.groupby(name)['Genus'].agg(['count', lambda x: x.mode()[0]]).rename(
            columns={'<lambda_0>': 'mode_cluster'})
        cluster_info['mode_size'] = summary_dataset.groupby([name, 'Genus']).size().groupby(level=0).max()
        cluster_info = cluster_info.merge(summary_dataset.groupby('Genus').size().rename('true_size'),
                                          left_on='mode_cluster', right_index=True)

        cluster_info['completeness'] = completeness
        cluster_info['homogeneity'] = homogeneity

        # Filter results for high quality clusters
        HQ = cluster_info[(cluster_info['completeness'] >= 0.50) & (cluster_info['homogeneity'] >= 0.50) & (
                    cluster_info['count'] >= 2)]
        GOOD_CLUSTERS[name] = HQ.index.values

        # Count unique high-quality clusters
        gt_counts = HQ['mode_cluster'].nunique()
        df2 = pd.DataFrame({dataset_column: [dataset_column], 'Count': [gt_counts]})
        print(df2)
        print(df2['Count'].sum())

        # Aggregate results for overall visualization
        overall_results[name] = pd.Series([completeness, homogeneity, gt_counts])

    # Visualization of overall results
    overall_results.T.plot(kind='bar', figsize=(10, 6), legend=True)
    plt.title("Cluster Quality Metrics Across Algorithms")
    plt.ylabel("Scores and Counts")
    plt.xlabel("Algorithms")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(f"Cluster_Quality_{dataset_column}.svg", format='svg', transparent=True)
    plt.show()

# Example usage
# analyze_clustering_results(summary_dataset, 'Environmental_Condition', algo_names)

