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
# FRAGMENT_LENGTHS = [10000, 50000, 100000, 250000, 500000, 1000000]
FRAGMENT_LENGTHS = [100000]
# PATH = "/home/m4safari/projects/def-lila-ab/m4safari/ext1prime/data"
PATH = "/content/drive/MyDrive/anew"
k = 8

def draw_fcgrs(sequence, id, len, domain, env_label, env):
    # Initialize FCGR object for (256x256) array representation
    fcgr = CGR_utils.FCGR(k=k, bits=8)
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
    plt.show()
    plt.tight_layout(pad=3.0)
    if not os.path.exists(f"../fcgrs/fragment_{len}/{k}/taxa/{domain}"):
        os.makedirs(f"../fcgrs/fragment_{len}/{k}/taxa/{domain}")
    plt.savefig(f"../fcgrs/fragment_{len}/{k}/taxa/{domain}/{id}.png", dpi=300)

    if not os.path.exists(f"../fcgrs/fragment_{len}/{k}/{env}/{env_label}"):
        os.makedirs(f"../fcgrs/fragment_{len}/{k}/{env}/{env_label}")
    plt.savefig(f"../fcgrs/fragment_{len}/{k}/{env}/{env_label}/{id}.png", dpi=300)

    if not os.path.exists(f"../fcgrs/fragment_{len}/{k}/all"):
        os.makedirs(f"../fcgrs/fragment_{len}/{k}/all")
    plt.savefig(f"../fcgrs/fragment_{len}/{k}/all/{id}.png", dpi=300)
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

def run(centroid):
    for env in ENVS:
        for fragment_length in FRAGMENT_LENGTHS:
            result_folder = f"../exp2/0/fragments_{fragment_length}"
            fasta_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')

            summary_path = f"../exp2/0/fragments_{fragment_length}/{env}/Extremophiles_{env}_Summary.tsv"
            summary_data = pd.read_csv(summary_path, sep='\t')


            id_2_sequences = read_fasta(fasta_file)
            for key  in centroid[env][fragment_length].keys():
                id = centroid[env][fragment_length][key]
                print(id)
                print(list(summary_data[summary_data['Assembly'] == id]))
                domain = str(list(summary_data[summary_data['Assembly'] == id]['Domain'])[0])
                env_label = str(list(summary_data[summary_data['Assembly'] == id][env])[0])
                print(f"Domain: {domain}, Env: {env_label}")
                sequence = clean_sequence(id_2_sequences[id])
                draw_fcgrs(sequence, id, fragment_length, domain, env_label, env)

    return id_2_sequences




centroid = {
    "pH":{
        100000 : {
       "Acidophiles": "GCA_000012285.1",
       "Alkaliphiles": "GCA_000255115.3"
    }
    },

    "Temperature": {
        100000 : {
       "Hyperthermophiles": "GCA_000258515.1",
       "Mesophiles": "GCA_000018465.1",
       "Psychrophiles": "GCA_000215995.1",
       "Thermophiles": "GCA_000007305.1"
    }
    }
}

id_2_sequences = run(centroid)

# result_folder = f"../exp2/0/fragments_{100000}"
# fasta_file = os.path.join(result_folder, "Temperature", f'Extremophiles_Temperature.fas')
# summary_path = f"../exp2/0/fragments_{100000}/Temperature/Extremophiles_Temperature_Summary.tsv"
# summary_data = pd.read_csv(summary_path, sep='\t')
# id_2_sequences = read_fasta(fasta_file)
#
# hyperthermophile = id_2_sequences["GCA_000007185.1"]
# thermophile = id_2_sequences["GCA_000008645.1"]
# joint_seq = hyperthermophile + "N" + thermophile
# print(len(joint_seq))
# draw_fcgrs(hyperthermophile, "GCA_000007185.1_Archea", len(hyperthermophile), "Archea", "hyperthermophile", "Temperature")
# draw_fcgrs(thermophile, "GCA_000008645.1_Archea", len(thermophile), "Archea", "thermophile", "Temperature")
# draw_fcgrs(joint_seq, "mixed", len(joint_seq), "Archea", "mixed", "Temperature")
#
#
# hyperthermophile = id_2_sequences["GCA_000016785.1"]
# thermophile = id_2_sequences["GCA_000020965.1"]
# joint_seq = hyperthermophile + "N" + thermophile
# print(len(joint_seq))
# draw_fcgrs(hyperthermophile, "GCA_000016785.1_Bacteria", len(hyperthermophile), "Bacteria", "hyperthermophile", "Temperature")
# draw_fcgrs(thermophile, "GCA_000020965.1_Bacteria", len(thermophile), "Bacteria", "thermophile", "Temperature")
# draw_fcgrs(joint_seq, "mixed", len(joint_seq), "Bacteria", "mixed", "Temperature")