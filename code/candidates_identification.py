from collections import Counter
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from Bio import SeqIO
import CGR_utils
import re
import numpy as np
from distance_calculator import descriptor_distance

GOOD_ALGO = {"Temperature":
                 ["VAE+IM", "VAE+affinity_propagation", "VAE+HDBSCAN", "UMAP+HDBSCAN", "iDeLUCS", "CL+HDBSCAN"],
             "pH":
                 ["VAE+IM", "VAE+affinity_propagation", "VAE+HDBSCAN", "iDeLUCS"]}

k = 8
ENVS = ["Temperature", "pH"]
ids_2_dist = {}

def candidates_identification(summary_dataset, env):
    pairs = {}
    file_path = os.path.join(f'/content/candidates_{env}.json')
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            pairs = json.load(file)

    for name in GOOD_ALGO[env]:
        assignment_algo = name

        for group in summary_dataset.groupby(assignment_algo):

            if (group[0] != -1):  # (group[0] in GOOD_CLUSTERS[name])
                species = group[1]["Species"].values
                ids = group[1]["Assembly"].values
                for i in range(len(species) - 1):
                    for j in range(i + 1, len(species)):
                        if group[1]["Domain"].values[i] != group[1]["Domain"].values[j]:
                            pair = tuple(sorted((species[i], species[j])))
                            pair_species = str(pair[0]) + "_" + str(pair[1])
                            if pair_species in pairs:
                                pairs[pair_species]["algo"].append(name)
                                pairs[pair_species]["pair_id_first"].append(ids[i])
                                pairs[pair_species]["pair_id_second"].append(ids[j])
                                pairs[pair_species]["counts"] += 1
                            else:
                                pairs[pair_species] = {}
                                pairs[pair_species]["pair_species_first"] = [species[i]]
                                pairs[pair_species]["pair_species_second"] = [species[j]]
                                pairs[pair_species]["algo"] = [name]
                                pairs[pair_species]["pair_id_first"] = [ids[i]]
                                pairs[pair_species]["pair_id_second"] = [ids[j]]
                                pairs[pair_species]["counts"] = 1

    with open(file_path, 'w') as file:
        json.dump(pairs, file, indent=2)


def analyse_candidates(file_dir, env):
    file_path = os.path.join(f'{file_dir}/candidates_{env}.json')

    with open(file_path, 'r') as file:
        pairs = json.load(file)

    final_pairs = {}
    just_pairs_ids = {}

    for i in pairs:
        clustering_models = list(set(pairs[i]["algo"]))
        count = pairs[i]["counts"]

        if len(clustering_models) >= 3 and count >= 5:
            final_pairs[i] = pairs[i]
            final_pairs[i]["algo"] = list(set(pairs[i]["algo"]))
            final_pairs[i]["pair_id_first"] = list(set(pairs[i]["pair_id_first"]))
            final_pairs[i]["pair_id_second"] = list(set(pairs[i]["pair_id_second"]))

            just_pairs_ids[final_pairs[i]["pair_species_first"][0]+","+final_pairs[i]["pair_species_second"][0]] = (final_pairs[i]["pair_id_first"][0], final_pairs[i]["pair_id_second"][0])

    print(len(just_pairs_ids))
    final_file_path = os.path.join(f'{file_dir}/final_candidates_{env}.json')
    with open(final_file_path, 'w') as file:
        json.dump(final_pairs, file, indent=2)

    final_file_path = os.path.join(f'{file_dir}/pairs_ids_candidates_{env}.json')
    with open(final_file_path, 'w') as file:
        json.dump(just_pairs_ids, file, indent=2)

    return just_pairs_ids

def read_fasta(file_path):

    id_2_sequences = {}
    # Open the FASTA file
    with open(file_path, 'r') as file:
        # Iterate over each record
        for record in SeqIO.parse(file, 'fasta'):
            id_2_sequences[str(record.id)] = str(record.seq)

    return id_2_sequences

def draw_fcgrs(sequence, id, domain, env_label, species, env):
    # Initialize FCGR object for (256x256) array representation
    fcgr = CGR_utils.FCGR(k=k, bits=8)
    # plt.figure(figsize=(12, 8))
    chaos_0 = fcgr(sequence[0])
    chaos_1 = fcgr(sequence[1])
    #
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(fcgr.plot(chaos_0), cmap="gray")
    axes[0].set_title(f"{domain[0]}_\n{env_label[0]}_\n{species[0]}", pad=20)
    axes[1].imshow(fcgr.plot(chaos_1), cmap="gray")
    axes[1].set_title(f"{domain[1]}_\n{env_label[1]}_\n{species[1]}", pad=20)
    # Title each subplot with letters
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_yticks([])

    ### distance calculation
    img1 = np.array(fcgr.plot(chaos_0))
    img2 = np.array(fcgr.plot(chaos_1))
    dist = descriptor_distance(img1, img2, 2, [0, 8, 16])
    ids_2_dist[f"{id[0]}_{id[1]}"] = dist


    # plt.tight_layout(pad=3.0)
    # if not os.path.exists(f"../candidates/fcgrs/{env}"):
    #     os.makedirs(f"../candidates/fcgrs/{env}")
    #
    # plt.savefig(f"../candidates/fcgrs/{env}/{id[0]}_{id[1]}.png", dpi=300)
    #

    # Coordinates of points to label (example points)
    # points = [(-3, 263), (-6, -5), (261, -5), (262, 262)]  # List of (x, y) tuples
    # labels = ["A", "C", "G", "T"]  # Labels for each point

    # Annotating each point
    # for (x, y), label in zip(points, labels):
    #     plt.text(x, y, label, color='black', fontsize=12, ha='center', va='center')
    plt.close()
    # plt.show()


def clean_sequence(sequence):
    # Replace any character not A, C, G, T, or N with N
    return re.sub(r'[^ACGTN]', 'N', sequence)
def run_FCGR(candidates, env):

    result_folder = f"../exp2/0/fragments_100000"
    fasta_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')

    summary_path = f"../exp2/0/fragments_100000/{env}/Extremophiles_{env}_Summary.tsv"
    summary_data = pd.read_csv(summary_path, sep='\t')
    id_2_sequences = read_fasta(fasta_file)

    for key in candidates.keys():
        domains = []
        env_labels = []
        sequences = []
        species = []
        id = [candidates[key][0], candidates[key][1]]

        domains.append(str(list(summary_data[summary_data['Assembly'] == id[0]]['Domain'])[0]))
        domains.append(str(list(summary_data[summary_data['Assembly'] == id[1]]['Domain'])[0]))

        env_labels.append(str(list(summary_data[summary_data['Assembly'] == id[0]][env])[0]))
        env_labels.append(str(list(summary_data[summary_data['Assembly'] == id[1]][env])[0]))

        sequences.append(clean_sequence(id_2_sequences[id[0]]))
        sequences.append(clean_sequence(id_2_sequences[id[1]]))

        keys = key.split(",")
        species.append(keys[0])
        species.append(keys[1])

        draw_fcgrs(sequences, id, domains, env_labels, species, env)

    return id_2_sequences


for env in ENVS:
    ids_2_dist = {}
    just_pairs_ids = analyse_candidates("../candidates",env)
    run_FCGR(just_pairs_ids, env)
    print(ids_2_dist)
    max_key = max(ids_2_dist, key=ids_2_dist.get)
    max_value = ids_2_dist[max_key]
    print(max_key, max_value)

    min_key = min(ids_2_dist, key=ids_2_dist.get)
    min_value = ids_2_dist[min_key]
    print(min_key, min_value)

    json_file_path = f'../Distances/candidates_{env}.json'
    # Write the dictionary to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(ids_2_dist, json_file, indent=4)