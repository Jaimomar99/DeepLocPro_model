from compute_dataset import PrecomputedCSVDataset
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from tqdm import tqdm
import numpy as np
import single_useful_functions as uf
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, help="Indicate the folder within DeepLocPro")
args = parser.parse_args()

path= ""
out_path = f"{path}/{args.output_folder}/"

emb_dir = f"{path}/data/single_embeddings_v2/"
data_file = f'{path}/data/single_label_dataset.csv'
part_file = f"{path}/data/single_graphpart.csv"


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
locations = ['Cellwall','Extracellular', 'Cytoplasmic', 'CYtoplasmicMembrane', 'OuterMembrane','Periplasmic']
device = uf.get_gpu()
uf.enforce_reproducibility()

best_models_param = pd.read_csv(f"{out_path}/metadata_single.csv", index_col=0)
outer_list = uf.outer_cross()

print("## Starting Testing ##")

for test, subset in outer_list:
    inner_list = uf.inner_cross(subset)
    _, lr, bs, drop = best_models_param.loc[f"test{test}"]
    test_dataset = PrecomputedCSVDataset(embeddings_dir=emb_dir, data_file= data_file, partitioning_file=part_file, partitions=[test])
    for validation, training in tqdm(inner_list):
        validation_dataset = PrecomputedCSVDataset(embeddings_dir=emb_dir, data_file= data_file, partitioning_file=part_file, partitions=[validation])
        test_loader= DataLoader(dataset=test_dataset, batch_size=int(bs), collate_fn=validation_dataset.collate_fn)
        model = uf.get_model(drop, device)
        model.load_state_dict(torch.load(f"{out_path}/models/model_{test}_{validation}", map_location=torch.device(device)))
        all_test_int, all_test_trues, all_test_kings = uf.test_loop(model, test_loader, device)
        pd.DataFrame(all_test_int, index=test_dataset.names, columns=locations).to_csv(f"{out_path}/probabilities/model_{test}_{validation}.csv")

        print(f"\t\tINNER: test:{test}, validation:{validation} run -- at {uf.get_time()}")
    #out of the inner because they are the same, better 5 dfs than 20
    pd.DataFrame(all_test_trues, index=test_dataset.names, columns=locations).to_csv(f"{out_path}/trues_labels/model_{test}.csv")
    pd.DataFrame(all_test_kings, index=test_dataset.names, columns=["kingdom"]).to_csv(f"{out_path}/kingdoms/model_{test}.csv")
    print(f"\t### \n\tOUTER: test:{test} done \n\t###")

print(" CROSS TEST DONE ")

#create the final 5 models
for test in range(5):
    probs = [f for f in os.listdir(f'{out_path}/probabilities/') if f.startswith(f"model_{test}")]
    dfs = []
    concatenated_df = pd.DataFrame()
    for file in probs:
        df = pd.read_csv(f"{out_path}/probabilities/{file}", index_col=0)
        dfs.append(df)
        concatenated_df = pd.concat(dfs, axis=1)

        average_df = pd.DataFrame()
    for location in locations:
        mean = concatenated_df.loc[:,location].mean(1)
        average_df[location] = mean
    average_df.to_csv(f"{out_path}/final_5_dfs/model_{test}.csv")

print("final 5 models created")

