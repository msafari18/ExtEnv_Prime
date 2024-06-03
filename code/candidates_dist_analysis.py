import matplotlib.pyplot as plt
import json
import pandas as pd
import os
from Bio import SeqIO


distance_metrics = ["descriptor", "dssim", "mldsp", "lpips"]
thresholds = [0.7, 0, 0.04, 0.6]


def plot_dist(dist_list, dist_metric):
    x_values = list(range(1, len(dist_list) + 1))
    # Create bar chart
    plt.bar(x_values, dist_list)
    plt.title(f'candidates bar chart plot/ temp / {dist_metric}')
    plt.xlabel('Index')
    plt.ylabel('distances')
    plt.grid(True)
    plt.show()


limited_dist = {}
for index ,d in enumerate(distance_metrics):
    limited_dist[d] = []
    with open(f"../Distances/{d}_candidates_dist_temp.json", 'r') as json_file:

        dist_data = json.load(json_file)
        for id in dist_data:
            if dist_data[id] <= thresholds[index]:
                limited_dist[d].append(id)

        print(d, len(limited_dist[d]))
        plot_dist(dist_data.values(), d)


for i in limited_dist:
    limited_dist[i] = set(limited_dist[i])

intersection_set = limited_dist[distance_metrics[0]]
for i in limited_dist:
    if i == "dssim":
        continue
    # print(limited_dist[i])
    intersection_set = intersection_set.intersection(limited_dist[i])

for i in intersection_set:
    print(i)




# import shutil
#
# file_names = []
# for i in list(intersection_set):
#     src = f"../candidates/fcgrs/all/{i}.png"
#     dst = f"../candidates/fcgrs/similar_3metrics/{i}.png"
#     shutil.copyfile(src, dst)

############################################################################################
# fragment_length = 100000
# env = "Temperature"
# sheet_name = "temp"
# # env = "pH"
#
# dist_metric = "MLDSP"
# file_path = "../Distances/dists100k6.xlsx"
# excel_data = pd.read_excel(file_path, engine='openpyxl', sheet_name=None)
#
# def read_fasta(file_path):
#
#     id_2_sequences = {}
#     # Open the FASTA file
#     with open(file_path, 'r') as file:
#         # Iterate over each record
#         for record in SeqIO.parse(file, 'fasta'):
#             id_2_sequences[str(record.id)] = str(record.seq)
#
#     return id_2_sequences
#
#
# def distance_matrix_to_dict(df):
#     dist_dict = {}
#     df.set_index('Unnamed: 0', inplace=True)
#     for row_label in df.index:
#         for col_label in df.columns[1:]:
#             if row_label != col_label:  # Avoid self-distances (if needed)
#                 row_label_l = row_label.replace("'", "")
#                 col_label_l = col_label.replace("'", "")
#
#                 dist_dict[row_label_l+"_"+col_label_l] = df.at[row_label, col_label]
#     return dist_dict
#
# sheet_names = list(excel_data.keys())
# distance_dicts_temp = {sheet_name: distance_matrix_to_dict(df) for sheet_name, df in excel_data.items()}
# distance_dicts = {}
# distance_dicts[sheet_name] = distance_dicts_temp[sheet_names[2]]
# # distance_dicts['pH'] =
#
# result_folder = f"../exp2/0/fragments_{fragment_length}"
# fasta_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')
#
# summary_path = f"../exp2/0/fragments_{fragment_length}/{env}/Extremophiles_{env}_Summary.tsv"
# summary_data = pd.read_csv(summary_path, sep='\t')
#
# id_2_sequences = read_fasta(fasta_file)
#
# # grouped_env = summary_data.groupby(env)
# grouped_env = summary_data.groupby("Domain")
#
# seq_ids = grouped_env["sequence_id"]
# seq_ids_dict = seq_ids.apply(list).to_dict()
#
# for key in seq_ids_dict.keys():
#     print("here")
#     print(key)
#
#     key_list = {}
#     for n1, id1 in enumerate(list(seq_ids_dict[key])):
#         for n2, id2 in enumerate(list(seq_ids_dict[key])):
#             if id2 == id1:
#                 continue
#             if id1+"_"+id2 in distance_dicts[sheet_name]:
#                 key_list[id1+"_"+id2] = distance_dicts[sheet_name][id1+"_"+id2]
#             else:
#                 key_list[id1 + "_" + id2] = distance_dicts[sheet_name][id2 + "_" + id1]
#
#     dist_list = key_list.values()
#     x_values = list(range(1, len(dist_list) + 1))
#     # Create bar chart
#     plt.bar(x_values, dist_list)
#     plt.title(f'bar chart / {key} / {dist_metric}')
#     plt.xlabel('Index')
#     plt.ylabel('distances')
#     plt.grid(True)
#     plt.show()
#     # plt.close()
