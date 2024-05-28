import math
import os
from PIL import Image
from skimage.metrics import structural_similarity, normalized_root_mse
import numpy as np
import CGR_utils
import matplotlib.pyplot as plt
import CGR_utils
from Bio import SeqIO
import os
import pandas as pd
import re
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

ENV = ['Temperature']

def split_image(image, split_size=14):
    h, w = image.shape[0], image.shape[1]
    col_count = int(math.ceil(h / split_size))
    row_count = int(math.ceil(w / split_size))
    tiles_count = col_count * row_count
    tiles = np.zeros((tiles_count, split_size, split_size))
    for y in range(col_count):
        for x in range(row_count):
            ind = x + (y * row_count)
            tiles[ind:(ind + 1), :, :] = image[split_size * y: (y + 1) * split_size,
                                         split_size * x:(x + 1) * split_size]

    return tiles


def get_descriptor(patch, bin_bounds):
    descriptor = np.zeros(len(bin_bounds))

    for index, bin_point in enumerate(bin_bounds):
        if index < len(bin_bounds) - 1:
            low = bin_bounds[index]
            high = bin_bounds[index + 1]
        else:
            low = bin_bounds[index]
            high = np.inf

        descriptor[index] = ((low <= patch) & (patch < high)).sum()
    descriptor = descriptor / np.sum(descriptor)
    descriptor = list(descriptor)
    return descriptor


def descriptor_distance(img1, img2, m=2, bins_bound=None):
    p1 = split_image(img1, 2 ** m)
    p2 = split_image(img2, 2 ** m)

    sub_matrices = p1.shape[0]

    vec1 = []
    vec2 = []
    for i in range(sub_matrices):
        vec1 += get_descriptor(patch=p1[i], bin_bounds=bins_bound)
        vec2 += get_descriptor(patch=p2[i], bin_bounds=bins_bound)

    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    denom_1 = np.sqrt(np.mean((vec1 * vec1), dtype=np.float64))
    denom_2 = np.sqrt(np.mean((vec2 * vec2), dtype=np.float64))
    if denom_1 > denom_2:
        distance = normalized_root_mse(vec1, vec2, normalization='euclidean')
    else:
        distance = normalized_root_mse(vec2, vec1, normalization='euclidean')

    # distance = normalized_root_mse(vec1, vec2, normalization='euclidean')
    # distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
    return distance

def fcgr_calculation(sequence):
    k = 8
    fcgr = CGR_utils.FCGR(k=k, bits=8)
    chaos = fcgr(sequence)
    img = np.array(fcgr.plot(chaos))

    return img



def read_fasta(file_path):

    id_2_sequences = {}
    # Open the FASTA file
    with open(file_path, 'r') as file:
        # Iterate over each record
        for record in SeqIO.parse(file, 'fasta'):
            id_2_sequences[str(record.id)] = str(record.seq)

    return id_2_sequences

def clean_sequence(sequence):
    # Replace any character not A, C, G, T, or N with N
    return re.sub(r'[^ACGTN]', 'N', sequence)


def compute_distance(id1, id2, seq_img_cache):
    img1 = seq_img_cache[id1]
    img2 = seq_img_cache[id2]
    dist = descriptor_distance(img1, img2, 2, [0, 8, 16])
    return f"{id1}_{id2}", dist


def run():
    # centroid = {
    #     "pH": {
    #             'Acidophiles': 'GCA_009729035.1',
    #             'Alkaliphiles': 'GCA_004745425.1',
    #     },
    #
    #     "Temperature": {
    #             'Hyperthermophiles': 'GCA_000258515.1',
    #             'Mesophiles': 'GCA_001006765.1',
    #             'Psychrophiles': 'GCA_008931805.1',
    #             'Thermophiles': 'GCA_009729035.1',
    #     }
    # }

    centroid = {
        "pH":{
            'Archaea':'GCA_000337135.1',
            'Bacteria': 'GCA_001447355.1',
        },

        "Temperature": {
            'Archaea': 'GCA_000012285.1',
            'Bacteria': 'GCA_008931805.1',
        }
    }

    dist_dict = {}
    fragment_length = 100000
    for env in ENV:

        result_folder = f"../exp2/0/fragments_{fragment_length}"
        fasta_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')

        summary_path = f"../exp2/0/fragments_{fragment_length}/{env}/Extremophiles_{env}_Summary.tsv"
        summary_data = pd.read_csv(summary_path, sep='\t')

        id_2_sequences = read_fasta(fasta_file)
        # print(id_2_sequences.keys())
        seq1 = id_2_sequences["GCA_000012285.1"]
        img1 = fcgr_calculation(seq1)

        seq2 = id_2_sequences["GCA_008931805.1"]
        img2 = fcgr_calculation(seq2)

        dist = descriptor_distance(img1, img2, 2, [0, 8, 16])
        print(dist)


    #     seq_img_cache = {seq_id: fcgr_calculation(id_2_sequences[seq_id]) for seq_id in id_2_sequences}
    #
    #     # grouped_env = summary_data.groupby(env)
    #     grouped_env = summary_data.groupby("Domain")
    #     seq_ids = grouped_env["sequence_id"]
    #     seq_ids_dict = seq_ids.apply(list).to_dict()
    #
    #
    #
    #     for key in seq_ids_dict.keys():
    #
    #         print(key)
    #         dist_dict[key] = {}
    #         id1 = centroid[env][key]
    #
    #         img1 = seq_img_cache[id1]
    #
    #         for _, id2 in enumerate(list(seq_ids_dict[key])):
    #             if id2 == id1:
    #                 continue
    #             img2 = seq_img_cache[id2]
    #
    #             dist = descriptor_distance(img1, img2, 2, [0, 8, 16])
    #             dist_dict[key][f"{id1}_{id2}"] = dist
    #
    #         ids_2_dist = dist_dict[key]
    #         max_key = max(ids_2_dist, key=ids_2_dist.get)
    #         max_value = ids_2_dist[max_key]
    #         print(max_key, max_value)
    #
    #         ids_2_dist = dist_dict[key]
    #         min_key = min(ids_2_dist, key=ids_2_dist.get)
    #         min_value = ids_2_dist[min_key]
    #         print(min_key, min_value)
    #
    #
    # return dist_dict


# dist_dict = run()
# for key in dist_dict.keys():
#     ids_2_dist = dist_dict[key]
#     max_key = max(ids_2_dist, key=ids_2_dist.get)
#     max_value = ids_2_dist[max_key]
#     print(max_key, max_value)
#
# json_file_path = f'../Distances/cluster_centroid_taxa.json'
# # Write the dictionary to the JSON file
# with open(json_file_path, 'w') as json_file:
#     json.dump(ids_2_dist, json_file, indent=4)

# if __name__ == '__main__':
#     img = np.arange(0, 64).reshape(8, 8)
#     img2 = np.arange(4, 68).reshape(8, 8)
#     dist = descriptor_distance(img, img2, 2, [0, 8, 16])
#     print(dist)



with open('../Distances/candidates_Temperature.json') as f:
    d = json.load(f)
    print(d)

import shutil

file_names = []
for i in d:
    if d[i] < 0.05:
        src = f"../candidates/fcgrs/all/{i}.png"
        dst = f"../candidates/fcgrs/similar_0.05/{i}.png"
        shutil.copyfile(src, dst)
        # file_names.append(f"{i}.json")



