import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
import sys

from sklearn.metrics import homogeneity_completeness_v_measure as cluster_quality
from models import CLE, VAE_encoder
from idelucs.cluster import iDeLUCS_cluster
from utils import kmersFasta

# Adding src to path for importing vamb
sys.path.append('src')
import vamb
vamb.parsecontigs()
vamb.parse_contigs()
# Setting matplotlib parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})



# Function to perform CLE (Contrastive Learning Encoder) encoding
def CL(sequence_file=None, latent_file=None):
    cl = CLE(sequence_file)
    names, latent = cl.encode()
    return names, latent


# Function to perform VAE (Variational Autoencoder) encoding
def VAE(sequence_file=None, latent_file=None):
    with open(sequence_file, 'rb') as contigfile:
        tnfs, names, contiglengths = vamb.parsecontigs.read_contigs(contigfile)

    # RPKM calculation (Replace with actual calculation if needed)
    rpkms = np.ones((tnfs.shape[0], 1), dtype=np.float32)

    # VAE model training and encoding
    vae = vamb.encode.VAE(nsamples=1, nlatent=32)
    dataloader, mask = vamb.encode.make_dataloader(rpkms, tnfs, batchsize=128)
    vae.trainmodel(dataloader)
    latent = vae.encode(dataloader)

    return names, latent


# Function to perform UMAP (Uniform Manifold Approximation and Projection) encoding
def UMAP(sequence_file=None, latent_file=None):
    names, kmers = kmersFasta(sequence_file, k=6)
    umap_model = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.0, n_components=64)
    return names, umap_model.fit_transform(kmers)


# Function to perform clustering using IM (Iterative Model)
def IM(latent, names):
    cluster_iterator = vamb.cluster.cluster(latent.astype(np.float32), labels=names)
    cluster_dict = {seq_id: c for c, seq_names in enumerate(cluster_iterator, 1) for seq_id in seq_names}
    y_pred = [cluster_dict[_id] for _id in names]
    return np.array(y_pred)


# Function to perform clustering using DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
def DBSCAN(latent, names):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True).fit(latent)
    return clusterer.labels_


# Function to insert clustering assignment into the dataset
def insert_assignment(summary_dataset, assignment_algo, GT_file, labels, level):
    summary_dataset[assignment_algo] = pd.Series(dtype='int')
    ground_truth_data = pd.read_csv(GT_file, sep='\t')

    for acc, label in zip(ground_truth_data["Assembly"].values, labels):
        summary_dataset.loc[summary_dataset["Assembly"] == acc, assignment_algo] = label

    GT = summary_dataset[level].to_numpy()
    unique_labels = np.unique(GT)
    y = np.array([unique_labels.tolist().index(x) for x in GT])
    y_pred = summary_dataset[assignment_algo].to_numpy()
    results = cluster_quality(y, y_pred)
    print(f"Results for {assignment_algo}: {results}")


def run(args):

    dataset = args.Env
    level = args.level
    path = args.results_folder

    sequence_file = f'{path}/{dataset}/Extremophiles_{dataset}.fas'
    GT_file = f'{path}/{dataset}/Extremophiles_{dataset}_GT_Tax.tsv'
    summary_file = f'{path}/Extremophiles_GTDB.tsv'
    # Loading dataset
    summary_dataset = pd.read_csv(summary_file, sep='\t',
                                  usecols=["Assembly", "Domain", dataset,
                                           "pH", "Phylum", "Class", "Order",
                                           "Family", "Genus", "Species"])
    summary_dataset.dropna(subset=[dataset], inplace=True)


    # Dimensionality reduction and clustering algorithms
    dim_algorithms = [("CL", CL), ("UMAP", UMAP)]
    clust_algorithms = [("IM", IM), ("HDBSCAN", DBSCAN)]

    # Running the algorithms
    for dim_name, dim_algo in dim_algorithms:
        names, latent = dim_algo(sequence_file=sequence_file)
        for clust_name, clust_algo in clust_algorithms:
            assignment_algo = f'{dim_name}+{clust_name}'
            print(assignment_algo)
            labels = clust_algo(latent, names)
            insert_assignment(summary_dataset, assignment_algo, GT_file, labels, level)

    # iDeLUCS Algorithm
    print("......... iDeLUCS ...............")
    params = {'sequence_file': sequence_file, 'n_clusters': 200, 'n_epochs': 50, 'n_mimics': 3,
              'batch_sz': 512, 'k': 6, 'weight': 0.25, 'n_voters': 5, 'optimizer': 'AdamW', 'scheduler': 'Linear'}
    model = iDeLUCS_cluster(**params)
    labels, latent = model.fit_predict(sequence_file)
    insert_assignment(summary_dataset, "iDeLUCS", GT_file, labels, level)

    env_gt_counts = {}
    GOOD_CLUSTERS = {}

    for env_value in summary_dataset[dataset].unique():
        env_subset = summary_dataset[summary_dataset[dataset] == env_value]
        gt_counts = env_subset[level].value_counts()
        gt_counts = gt_counts[gt_counts >= 2]
        env_gt_counts[env_value] = len(gt_counts)

    df = pd.DataFrame(env_gt_counts.items(), columns=[dataset, 'True Genera'])
    print(df)

    algo_names = []
    for dim_name, dim_algo in dim_algorithms:
        for clust_name, clust_algo in clust_algorithms:
            algo_names.append(f'{dim_name}+{clust_name}')

    algo_names.append('iDeLUCS')
    # algo_names = ["CL+HDBSCAN", "VAE+HDBSCAN", "UMAP+HDBSCAN"]

    for name in algo_names:
        assignment_algo = name
        print(assignment_algo)

        results_df = pd.DataFrame()
        results_df[assignment_algo] = summary_dataset[assignment_algo]
        results_df['true_genus'] = summary_dataset[level]
        results_df['mode_cluster'] = summary_dataset.groupby(assignment_algo)[level].transform(
            lambda x: x.value_counts().idxmax())
        results_df['cluster_size'] = summary_dataset.groupby(assignment_algo)[level].transform(lambda x: x.count())

        results_df["true_size"] = pd.Series(dtype='int')
        genus_size = dict(summary_dataset.groupby(level)[level].count())
        for i, acc in enumerate(results_df["mode_cluster"].values):
            results_df.loc[results_df["mode_cluster"] == acc, "true_size"] = genus_size[acc]

        results_df["mode_size"] = pd.Series(dtype='int')
        mode_size = dict(summary_dataset[results_df.mode_cluster == results_df.true_genus].groupby(assignment_algo)[
                             assignment_algo].count())
        for i, acc in enumerate(results_df[assignment_algo].values):
            results_df.loc[results_df[assignment_algo] == acc, "mode_size"] = mode_size[acc]

        del results_df["true_genus"]

        results_df["completeness"] = np.minimum(results_df["mode_size"].values, results_df["true_size"].values) / \
                                     results_df["true_size"].values
        results_df["contamination"] = (results_df["cluster_size"].values - results_df["mode_size"].values) / results_df[
            "cluster_size"].values
        # results_df[dataset] = pd.Series(dtype='int')
        results_df[dataset] = results_df['mode_cluster'].map(
            summary_dataset.drop_duplicates(subset=[level]).set_index(level)[dataset])
        results_df[dataset] = summary_dataset.groupby(assignment_algo)[dataset].transform(
            lambda x: x.value_counts().idxmax())

        clustering_results = results_df.drop_duplicates()

        clustering_results = clustering_results.loc[clustering_results['cluster_size'] >= 2]

        # Group the rows by "Env" and count the number of unique values in "GT"
        HQ = clustering_results[
            (clustering_results["completeness"] >= 0.50) & (clustering_results["contamination"] <= 0.50)]
        GOOD_CLUSTERS[name] = HQ[name].values
        gt_counts = HQ.groupby(dataset)["mode_cluster"].nunique()
        df2 = pd.DataFrame({dataset: gt_counts.index, 'Count': gt_counts.values})
        print(df2)
        print(df2['Count'].sum())

        df[name] = np.zeros(len(df))
        look = dict(zip(df2[dataset], df2["Count"]))
        for key in look:
            df.loc[df[dataset] == key, name] = look[key]

    # Plot the merged dataframe
    df.plot.bar(x=dataset, rot=0, figsize=(8, 6.8), legend=True, width=0.4, fontsize=14)
    plt.title("No. True Genera vs. No. Recovered Genera by each Algorithm", fontsize=16)
    plt.ylabel("No. Genera", fontsize=14)
    plt.ylim(0, 37)
    plt.xlabel(dataset, fontsize=14)

    # Saving and displaying the plot
    plt.savefig(f"Unsupervised_{dataset}.svg", format="svg", transparent=True)
    plt.show()

    if dataset=="Temperature":
        algo_names = ["VAE+HDBSCAN", "CL+HDBSCAN", "VAE+IM", "UMAP+HDBSCAN","iDeLUCS"]
    else:
        algo_names = ["VAE+HDBSCAN", "CL+HDBSCAN", "VAE+IM", "CL+IM" ,"iDeLUCS"]
    #algo_names = ["VAE+HDBSCAN", "VAE+IM","iDeLUCS"]

    pairs={}
    for name in algo_names:
        assignment_algo = name

        for group in summary_dataset.groupby(assignment_algo):
            #print(group[0])
            if ( group[0] != -1):   #(group[0] in GOOD_CLUSTERS[name])
                species = group[1]["Species"].values
                for i in range(len(species)-1):
                    for j in range(i+1, len(species)):
                        if group[1]["Domain"].values[i] != group[1]["Domain"].values[j]:
                            if tuple(sorted((species[i], species[j]))) in pairs:
                                pairs[tuple(sorted((species[i], species[j])))].append(name)
                            else:
                                pairs[tuple(sorted((species[i], species[j])))] = [name]
                            #print(species[i], species[j])

    L = []
    for pair in pairs:
        if len(pairs[pair])>=2:
            print(pair, pairs[pair])
            L.append(pair[0].replace(' ','_'))
            L.append(pair[1].replace(' ','_'))


def main():

    # Argument Parser
    parser = argparse.ArgumentParser(description="Run genomic clustering algorithms.")

    parser.add_argument('--results_folder', action='store', type=str)
    parser.add_argument('--Env', type=str, default="Temperature", help='Dataset to use, e.g., "Temperature", "pH".')
    parser.add_argument('--level', type=str, default="Genus", help='Taxonomic level, e.g., "Genus".')
    # parser.add_argument('--summary_file', type=str, required=True, help='Path to the summary file.')
    # parser.add_argument('--sequence_file', type=str, required=True, help='Path to the sequence file.')
    # parser.add_argument('--GT_file', type=str, required=True, help='Path to the ground truth file.')
    args = parser.parse_args()

    args = vars(parser.parse_args())

    run(args)

if __name__ == '__main__':
    main()