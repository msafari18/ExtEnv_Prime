cd Extreme_Env/code
# Set the path and parameters
path="/home/m4safari/projects/def-lila-ab/m4safari/ext1prime/data"
fragment_sizes=(100000 250000 500000 1000000)



for fragment_size in "${fragment_sizes[@]}"; do
    fragment_path="challengingexp/fragments_${fragment_size}"

    # Run the Build Signature Dataset Python script
    python3 /home/m4safari/projects/def-lila-ab/m4safari/ext1prime/Extreme_Env/code/Build_Signature_Dataset.py \
        --path "$path" \
        --dataset_file "Extremophiles_GTDB.tsv" \
        --fragment_len "$fragment_size" \
        --new_folder_name "$fragment_path"
    echo "new fragments with size ${fragment_size} created"

    result_path="${path}/challengingexp/fragments_${fragment_size}"

    # Run the Supervised Models Python script for Temperature
    python3 /home/m4safari/projects/def-lila-ab/m4safari/ext1prime/Extreme_Env/code/SupervisedModels_Challenging.py \
        --results_folder="$result_path" \
        --Env="Temperature" \
        --n_clusters=4 \
        --exp "$exp" \
        --max_k 9

    # Run the Supervised Models Python script for pH
    python3 /home/m4safari/projects/def-lila-ab/m4safari/ext1prime/Extreme_Env/code/SupervisedModels_Challenging.py \
        --results_folder="$result_path" \
        --Env="pH" \
        --n_clusters=2 \
        --exp "$exp" \
        --max_k 9

echo "EXPERIMENT DONE"

done
