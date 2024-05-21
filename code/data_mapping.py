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
    'Genome Size (Mb)': 'genome_size (M)',
    'Domain': 'Domain',
    'Species': 'Species',
    'Reference': 'RefLink',
    'Radio Label': 'Radio Label'
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
print(new_radiophiles_df.head())

# Save the new dataframe to a CSV file if needed
# new_radiophiles_df.to_csv('../data/Extremophiles_GTDB_Radio.tsv', sep='\t', index=False)

directory_path = "../data/ncbi-genomes-2024-05-15"
all_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

found = []
for i in list(new_radiophiles_df["Assembly"]):
    for k in all_files:
        name = i.strip()
        if name in k:
            found.append(name)
            os.mkdir(f"../data/Assembly/{name}")
            source_file = f"../data/ncbi-genomes-2024-05-15/{k}"
            destination_file = f"../data/Assembly/{name}/{k}"
            shutil.copy(source_file, destination_file)

for i in list(new_radiophiles_df["Assembly"]):
    if i not in found:
        print(i)
