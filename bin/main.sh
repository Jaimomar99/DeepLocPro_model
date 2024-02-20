#!/bin/bash

#GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#modify
output_folder="results/"
epochs=60

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

nohup python ./bin/Training_validation_cross.py\
    --output_folder ${output_folder}\
    --epochs ${epochs}\
    > ${path}${output_folder}/logs/Training_validation_${DATE}.log 2>&1

nohup python ${path}/bin/Test_cross.py\
    --output_folder ${output_folder}\
    > ${path}${output_folder}/logs/Test_cross${DATE}.log 2>&1

nohup python ${path}/bin/Metrics.py\
    --output_folder ${output_folder}\
    > ${path}${output_folder}/logs/Metrics${DATE}.log 2>&1