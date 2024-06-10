import pandas as pd
import matplotlib.pyplot as plt

# Define function to load and analyze feature importance files
def analyze_feature_importance(file_paths):
    results = []

    for file_path in file_paths:
        # Load the CSV file
        df = pd.read_csv(file_path)
        columns_to_convert = [i for i in df.columns[1:]]
        # print(df.columns)
        df[columns_to_convert] = df[columns_to_convert].astype(float)

        # Check the structure of the DataFrame
        print(f"Analyzing {file_path}")
        # print(df)

        # Identify k-mers with positive or negative effects
        positive_effects = (df[columns_to_convert] > 0).sum(axis=1)
        negative_effects = (df[columns_to_convert] < 0).sum(axis=1)
        print(positive_effects)
        # Find k-mers that have positive or negative effects in at least 3 metrics
        significant_kmers = df[(positive_effects >= 3)]

        # Store results
        results.append(significant_kmers)


    # Combine results from all files
    combined_results = pd.concat(results)

    return combined_results


# List of file paths

# for i in file_paths:
#     df = pd.read_csv(i)
#     df['perm_svm'].plot()
#     plt.show()
# Analyze feature importance files
# combined_results = analyze_feature_importance(file_paths)

# Display the combined results
# print(combined_results)


file_paths = {
'pH_env': '../k_mer_importance/feature_importances_pH_Env.csv',
'pH_taxa': '../k_mer_importance/feature_importances_pH_Tax.csv',
'Temperature_env': '../k_mer_importance/feature_importances_Temperature_Env.csv',
'Temperature_taxa': '../k_mer_importance/feature_importances_Temperature_Tax.csv'
}


def load_and_plot_feature_importances(df, name):


    # Assuming k-mers are in the first column and setting it as the index
    if df.columns[0] != 'k-mers':
        df.rename(columns={df.columns[0]: 'k-mers'}, inplace=True)
    df.set_index('k-mers', inplace=True)

    # Compute mean importance of each k-mer across all metrics
    feature_importance = df["mdi"]

    # Sort the features by importance
    feature_importance = feature_importance.sort_values(ascending=False)
    print(name)
    print(list(feature_importance.keys()[:15]))
    # Plotting
    plt.figure(figsize=(12, 8))
    feature_importance.plot(kind='bar')
    plt.title(f'Feature Importance - {name}')
    plt.xlabel('k-mers')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)  # Rotate k-mer labels for better visibility
    plt.tight_layout()
    plt.show()

dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Plot feature importances for each file
for name, df in dataframes.items():
    load_and_plot_feature_importances(df, name)