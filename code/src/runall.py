import os
import glob

path = "/content/drive/MyDrive/University_of_Waterloo/Extremophile_1prime/data"
fragment_sizes = [100000]
experiments_num = [i for i in range(1,10)]


for exp in experiments_num:

  for fragment_size in fragment_sizes:

    fragment_path = f'fragments_{fragment_size}'

    !python3 /content/Extreme_Env/code/Build_Signature_Dataset.py \
      --path $path \
      --dataset_file Extremophiles_GTDB.tsv \
      --fragment_len $fragment_size \
      --new_folder_name $fragment_path

    print(f"new fragments with size {fragment_size} created")

    result_path = f'{path}/fragments_{fragment_size}'
    !python3 SupervisedModels.py \
    --results_folder=$result_path \
    --Env=Temperature \
    --n_clusters=4 \
    --exp $exp \
    --max_k 9 \


    result_path = f'{path}/fragments_{fragment_size}'
    python3 SupervisedModels.py \
    --results_folder=$result_path \
    --Env=pH \
    --n_clusters=2 \
    --exp $exp \
    --max_k 9 \


  # f'{path}/{fragment_path}/Temperature/Extremophiles_Temperature_Summary.tsv'
    rm_path = [ f'{path}/{fragment_path}/Temperature/Extremophiles_Temperature_Summary.tsv',
            f'{path}/{fragment_path}/Temperature/Extremophiles_Temperature_GT_Tax.tsv',
            f'{path}/{fragment_path}/Temperature/Extremophiles_Temperature_GT_Env.tsv',
            f'{path}/{fragment_path}/Temperature/Extremophiles_Temperature.fas',
            f'{path}/{fragment_path}/pH/Extremophiles_pH_Summary.tsv',
            f'{path}/{fragment_path}/pH/Extremophiles_pH_GT_Tax.tsv',
            f'{path}/{fragment_path}/pH/Extremophiles_pH_GT_Env.tsv',
            f'{path}/{fragment_path}/pH/Extremophiles_pH.fas'
    ]
    for file in rm_path:
      os.remove(file)

  print(f"{exp+1} EXPERIMENT DONE")

if [ ! -d "$dir_path" ]; then
    # The directory does not exist, so create it
    mkdir -p "$dir_path"
    echo "Directory created: $dir_path"
else
    echo "Directory already exists: $dir_path"
fi



