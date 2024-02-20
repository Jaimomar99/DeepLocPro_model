#!/bin/bash

#GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#modify
output_folder="results/"
epochs=5

DATE=`date +"%Y-%m-%d_%b"`

if [ ! -d "${output_folder}" ]; then
    mkdir -p "${output_folder}"
fi

subdirectories=("final_5_dfs/" "kingdoms/" "models/" "probabilities/" "trues_labels/" "logs/")
for d in "${subdirectories[@]}"; do
    if [ ! -d "${output_folder}${d}" ]; then
    mkdir -p "${output_folder}${d}"
    fi
done

echo ${PWD}

nohup python ./bin/make_embeddings_fsdp_v2.py\
    --fasta_file ./data/example.fasta\
    --output_embeddings ./data/embeddings\
    > ./${output_folder}/logs/Embeddings_sequences${DATE}.log 2>&1

nohup python ./bin/Training_validation_cross.py\
    --output_folder ${output_folder}\
    --epochs ${epochs}\
    > ./${output_folder}/logs/Training_validation_${DATE}.log 2>&1

nohup python ./bin/Test_cross.py\
    --output_folder ${output_folder}\
    > ./${output_folder}/logs/Test_cross${DATE}.log 2>&1

nohup python .//bin/Metrics.py\
    --output_folder ${output_folder}\
    > ./${output_folder}/logs/Metrics${DATE}.log 2>&1