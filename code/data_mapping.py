import pandas as pd
import os
import shutil

# Load the second file
file_path_2 = '../data/radiophiles.xlsx'
radiophiles_data = pd.read_excel(file_path_2)

# Load the first file to get its column structure
extremophiles_df = pd.read_csv('../data/Extremophiles_GTDB.tsv', sep='\t')

# Define the column mapping
column_mapping = {
    'Accession Number': 'Assembly',
    'Genome Size (Mb)': 'genome_size',
    'Domain': 'Domain',
    'Species': 'Species',
    'Genus': 'Genus',
    'Reference': 'RefLink',
    'Radio_Label': 'Radio_label'
}

# Initialize the new dataframe with the same columns as the first file
new_columns = list(extremophiles_df.columns)
new_columns.remove('Temperature')  # Remove 'Temperature' column
new_columns.remove('pH')  # Remove 'pH' column
new_columns.append('Radio Label')  # Add 'Radio Label' column

# Create an empty dataframe with the new columns
new_radiophiles_df = pd.DataFrame(columns=new_columns)

# Fill in the data based on the mapping
for old_col, new_col in column_mapping.items():
    new_radiophiles_df[new_col] = radiophiles_data[old_col]

# Fill the remaining columns with NaN
remaining_columns = set(new_columns) - set(column_mapping.values())
for col in remaining_columns:
    new_radiophiles_df[col] = float('nan')

# Display the first few rows of the new dataframe to ensure correctness
# print(new_radiophiles_df.head())
#
# assembly_2_genus = {}
# for id, g in zip(new_radiophiles_df["Assembly"], new_radiophiles_df["Genus"]):
#     assembly_2_genus[id] = g
#
# for length in [10000, 50000, 100000, 250000, 500000, 1000000]:
#     path = f"../data/exp2/0/fragments_{length}/Radio Label/Extremophiles_Radio Label_Summary.tsv"
#     data = pd.read_csv(path, sep='\t')
#     genus = []
#     for i in data["Assembly"]:
#         genus.append(assembly_2_genus[i])
#     data['genus'] = genus
#     data.to_csv(path)


# Save the new dataframe to a CSV file if needed


# directory_path = "../data/ncbi-genomes-2024-05-15"
# all_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
#
# found = []
# for i in list(new_radiophiles_df["Assembly"]):
#     for k in all_files:
#         name = i.strip()
#         if name in k:
#             found.append(name)
#             os.mkdir(f"../data/Assembly/{name}")
#             source_file = f"../data/ncbi-genomes-2024-05-15/{k}"
#             destination_file = f"../data/Assembly/{name}/{k}"
#             shutil.copy(source_file, destination_file)
#
new_radiophiles_df["Assembly"] = [i.strip() for i in new_radiophiles_df["Assembly"]]
# for i in list(new_radiophiles_df["Assembly"]):
#     if i not in found:
#         print(i)

new_radiophiles_df.to_csv('../data/Extremophiles_GTDB_Radio.tsv', sep='\t', index=False)