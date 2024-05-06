"""Generating avoided pattern plot for the paper"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import CGR_utils
from Bio import SeqIO
import os
import pandas as pd
import re

ENVS = ["Temperature", "pH"]
NUM_CLUSTERS = {"Temperature": 4, "pH": 2}
FRAGMENT_LENGTHS = [10000, 50000, 100000, 250000, 500000, 1000000]
FRAGMENT_LENGTHS = [100000]
# PATH = "/home/m4safari/projects/def-lila-ab/m4safari/ext1prime/data"
PATH = "/content/drive/MyDrive/anew"


def draw_fcgrs(sequence, id, len, domain, env_label, env):
    # Initialize FCGR object for (256x256) array representation
    fcgr = CGR_utils.FCGR(k=8, bits=8)
    # plt.figure(figsize=(12, 8))
    chaos = fcgr(sequence)
    plt.imshow(fcgr.plot(chaos), cmap="gray")
    # Title each subplot with letters
    plt.title(f"{id}", y=-0.1)
    plt.xticks([])  # Remove x ticks
    plt.yticks([])  # Remove y ticks
    # Coordinates of points to label (example points)
    points = [(-3, 263), (-6, -5), (261, -5), (262, 262)]  # List of (x, y) tuples
    labels = ["A", "C", "G", "T"]  # Labels for each point

    # Annotating each point
    for (x, y), label in zip(points, labels):
        plt.text(x, y, label, color='black', fontsize=12, ha='center', va='center')
    plt.tight_layout(pad=3.0)
    if not os.path.exists(f"{PATH}/fcgrs/fragment_{len}/taxa/{domain}"):
        os.makedirs(f"{PATH}/fcgrs/fragment_{len}/taxa/{domain}")
    plt.savefig(f"{PATH}/fcgrs/fragment_{len}/taxa/{domain}/{id}.png", dpi=300)

    if not os.path.exists(f"{PATH}/fcgrs/fragment_{len}/{env}/{env_label}"):
        os.makedirs(f"{PATH}/fcgrs/fragment_{len}/{env}/{env_label}")
    plt.savefig(f"{PATH}/fcgrs/fragment_{len}/{env}/{env_label}/{id}.png", dpi=300)

    if not os.path.exists(f"{PATH}/fcgrs/fragment_{len}/all"):
        os.makedirs(f"{PATH}/fcgrs/fragment_{len}/all")
    plt.savefig(f"{PATH}/fcgrs/fragment_{len}/all/{id}.png", dpi=300)
    plt.close()

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

def run():
    for env in ENVS:
        for fragment_length in FRAGMENT_LENGTHS:
            result_folder = f"{PATH}/exp1/0/fragments_{fragment_length}"
            fasta_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')

            summary_path = f"{PATH}/exp1/0/fragments_{fragment_length}/{env}/Extremophiles_{env}_Summary.tsv"
            summary_data = pd.read_csv(summary_path, sep='\t')


            id_2_sequences = read_fasta(fasta_file)
            for id in id_2_sequences:
                domain = str(list(summary_data[summary_data['Assembly'] == id]['Domain'])[0])
                env_label = str(list(summary_data[summary_data['Assembly'] == id][env])[0])
                print(f"Domain: {domain}, Env: {env_label}")
                sequence = clean_sequence(id_2_sequences[id])
                draw_fcgrs(sequence, id, fragment_length, domain, env_label, env)





