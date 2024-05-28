import pandas as pd
import json
import os
import CGR_utils
from Bio import SeqIO
from distance_calculator import read_fasta
import matplotlib.pyplot as plt

file_path = '../Distances/dssim_distances_pH.xlsx'

centroid_env = {
    "pH":{
       'Acidophiles': 'GCA_009729035.1',
       'Alkaliphiles': 'GCA_004745425.1',
    },

    "Temperature": {
        'Hyperthermophiles': 'GCA_000258515.1',
        'Mesophiles':'GCA_001006765.1',
        'Psychrophiles':'GCA_008931805.1',
        'Thermophiles':'GCA_009729035.1',
    }
}

centroid_taxa = {
    "pH":{
            'Archaea':'GCA_000337135.1',
            'Bacteria': 'GCA_001447355.1',
    },

    "Temperature": {
        'Archaea': 'GCA_000012285.1',
        'Bacteria': 'GCA_008931805.1',
    }
}

excel_data = pd.read_excel(file_path, engine='openpyxl', sheet_name=None)

# Display the names of the sheets and the first few rows of each sheet to understand its structure
sheet_names = list(excel_data.keys())
sheets_preview = {sheet: excel_data[sheet].head() for sheet in sheet_names}


# Function to convert a distance matrix DataFrame to a dictionary with pairs as keys
def distance_matrix_to_dict(df):
    dist_dict = {}
    df.set_index('Unnamed: 0', inplace=True)
    for row_label in df.index:
        for col_label in df.columns[1:]:
            if row_label != col_label:  # Avoid self-distances (if needed)
                row_label_l = row_label.replace("'", "")
                col_label_l = col_label.replace("'", "")

                dist_dict[row_label_l+"_"+col_label_l] = df.at[row_label, col_label]
    return dist_dict


# Convert each sheet to a dictionary
distance_dicts = {sheet_name: distance_matrix_to_dict(df) for sheet_name, df in excel_data.items()}
distance_dicts['pH'] = distance_dicts['Sheet1']
######################### distances for candidates
# path = '../Distances/candidates_pH.json'
# # path = '../Distances/candidates_Temperature.json'
# #
# with open(path) as f:
#     ids_dict = json.load(f)
#
#
# gurjit_dist = {}
# gurjit_dist_list = []
#
# for id in ids_dict.keys():
#     gurjit_dist[id] = distance_dicts['pH'][id]
#     gurjit_dist_list.append(distance_dicts['pH'][id])
#
# max_key = max(gurjit_dist, key=gurjit_dist.get)
# max_value = gurjit_dist[max_key]
# print(max_key, max_value)
#
# max_key = min(gurjit_dist, key=gurjit_dist.get)
# max_value = gurjit_dist[max_key]
# print(max_key, max_value)


# x_values = list(range(1, len(gurjit_dist_list) + 1))
#
# # Create bar chart
# plt.bar(x_values, gurjit_dist_list)
# # Create scatter plot
# # plt.scatter(x_values, gurjit_dist_list)
# plt.title('candidates bar chart plot/ pH / DSSIM')
# plt.xlabel('Index')
# plt.ylabel('distances')
# plt.grid(True)
# plt.show()



############################# distances vs distances

# t_a = centroid_taxa["Temperature"]['Archaea']
# t_b = centroid_taxa["Temperature"]['Bacteria']
# p_a = centroid_taxa["pH"]['Archaea']
# p_b = centroid_taxa["pH"]['Bacteria']
#
# # a_b_distance_temp = distance_dicts['temp'][t_a+"_"+t_b]
# a_b_distance_pH = distance_dicts['pH'][p_a+"_"+p_b]
# print(a_b_distance_pH)

############################ distances withing the clusters
fragment_length = 100000
# env = "Temperature"
env = "pH"

result_folder = f"../exp2/0/fragments_{fragment_length}"
fasta_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')

summary_path = f"../exp2/0/fragments_{fragment_length}/{env}/Extremophiles_{env}_Summary.tsv"
summary_data = pd.read_csv(summary_path, sep='\t')

id_2_sequences = read_fasta(fasta_file)

# grouped_env = summary_data.groupby(env)
grouped_env = summary_data.groupby("Domain")
seq_ids = grouped_env["sequence_id"]
seq_ids_dict = seq_ids.apply(list).to_dict()

for key in seq_ids_dict.keys():
    print("here")
    print(key)

    key_list = {}
    for n1, id1 in enumerate(list(seq_ids_dict[key])):
        for n2, id2 in enumerate(list(seq_ids_dict[key])):
            if id2 == id1:
                continue
            if id1+"_"+id2 in distance_dicts['pH']:
                key_list[id1+"_"+id2] = distance_dicts['pH'][id1+"_"+id2]
            else:
                key_list[id1 + "_" + id2] = distance_dicts['pH'][id2 + "_" + id1]

    max_key = max(key_list, key=key_list.get)
    max_value = key_list[max_key]
    print("max: ",max_key, max_value)

    max_key = min(key_list, key=key_list.get)
    max_value = key_list[max_key]
    print("min: ",max_key, max_value)


