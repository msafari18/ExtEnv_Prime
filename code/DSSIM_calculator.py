import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import CGR_utils
from Bio import SeqIO
import math
from skimage.metrics import structural_similarity, normalized_root_mse
import numpy as np


ENVS = ['Temperature', 'pH']
FRAGMENT_LENGTHS = [100000]

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


def calculate_dssim(image1, image2):
    # Ensure the images are the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for DSSIM calculation")

    # Calculate SSIM
    ssim_index, _ = ssim(image1, image2, full=True)

    # Calculate DSSIM
    dssim_index = (1 - ssim_index) / 2

    return dssim_index

def read_fasta(file_path):

    id_2_sequences = {}
    # Open the FASTA file
    with open(file_path, 'r') as file:
        # Iterate over each record
        for record in SeqIO.parse(file, 'fasta'):
            id_2_sequences[str(record.id)] = str(record.seq)

    return id_2_sequences

def creat_images():
    envs_dict = []
    for env in ENVS:
        sequences = {}
        for fragment_length in FRAGMENT_LENGTHS:
            result_folder = f"../exp2/0/fragments_{fragment_length}"
            fasta_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')

            id_2_sequences = read_fasta(fasta_file)
            for id in id_2_sequences:
                seq = id_2_sequences[id]
                img = fcgr_calculation(seq)
                sequences[id] = img

        envs_dict.append(sequences)

    return envs_dict




def create_distance_matrix(images):
    image_names = list(images.keys())
    distance_matrix = pd.DataFrame(index=image_names, columns=image_names)

    for i in tqdm(range(len(image_names))):
        for j in range(i, len(image_names)):
            if i == j:
                distance_matrix.at[image_names[i], image_names[j]] = 0.0
            else:
                # dssim_value = calculate_dssim(images[image_names[i]], images[image_names[j]])
                dssim_value = descriptor_distance(images[image_names[i]], images[image_names[j]], 2, [0, 8, 16])
                # dssim_value = calculate_dssim(images[image_names[i]], images[image_names[j]])
                distance_matrix.at[image_names[i], image_names[j]] = dssim_value
                distance_matrix.at[image_names[j], image_names[i]] = dssim_value

    return distance_matrix

# Load the images
sequences = creat_images()
print("done 1")
# Create the DSSIM distance matrix
distance_matrix = create_distance_matrix(sequences[0])
# distance_matrix = create_distance_matrix(sequences[1])

# Save the distance matrix to an Excel file
output_file_path = '../Distances/des_distances_temp.xlsx'
distance_matrix.to_excel(output_file_path)

print(f"descriptor distance matrix saved to {output_file_path}")
