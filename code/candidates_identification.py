from collections import Counter

import os
import json

GOOD_ALGO = {"Temperature":
                 ["VAE+IM", "VAE+affinity_propagation", "VAE+HDBSCAN", "UMAP+HDBSCAN", "iDeLUCS", "CL+HDBSCAN"],
             "pH":
                 ["VAE+IM", "VAE+affinity_propagation", "VAE+HDBSCAN", "iDeLUCS"]}


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
