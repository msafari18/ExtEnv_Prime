import pandas as pd


file_path = "../exp2/0/fragments_10000/Temperature/Extremophiles_Temperature_Summary.tsv"

data = pd.read_csv(file_path, sep="\t")

temperature_counts = data.groupby(['Domain', 'Temperature'])['genus'].unique()

for i in temperature_counts:
    print(len(i))
