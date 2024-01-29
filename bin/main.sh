#!/bin/bash

#run in terminal
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#modify
path="/home/zxv353/isdata_zxv353/ProDeepLoc/"
output_folder="results_softmax_without_v2/"
epochs=60

DATE=`date +"%Y-%m-%d_%b"`

out_path="${path}/${output_folder}/"
if [ ! -d "${out_path}" ]; then
    mkdir -p "${out_path}"
fi

subdirectories=("final_5_dfs/" "kingdoms/" "models/" "probabilities/" "trues_labels/" "logs/")
for d in "${subdirectories[@]}"; do
    if [ ! -d "${out_path}${d}" ]; then
    mkdir -p "${out_path}${d}"
    fi
done

# nohup python ${path}/bin/Training_validation_cross.py\
#     --output_folder ${output_folder}\
#     --epochs ${epochs}\
#     > ${path}${output_folder}/logs/Training_validation_${DATE}.log 2>&1

# nohup python ${path}/bin/Test_cross.py\
#     --output_folder ${output_folder}\
#     > ${path}${output_folder}/logs/Test_cross${DATE}.log 2>&1

nohup python ${path}/bin/Metrics.py\
    --output_folder ${output_folder}\
    > ${path}${output_folder}/logs/Metrics${DATE}.log 2>&1