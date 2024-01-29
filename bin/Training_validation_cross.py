import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import single_useful_functions as uf
import pandas as pd
import warnings
from sklearn.metrics import  f1_score, accuracy_score
import os
import argparse

#instructions:
'''
Full cross validation
For all hyper combinations, we run the 4 models, cross validating, and takingthe  average of F1 of the validation.
and take best hyperparameter combination, keep the 4 models.
'''

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, help="Indicate the folder within ProDeepLoc")
parser.add_argument("--epochs", type=str, help="Indicate the number of epochs for training and validation")
args = parser.parse_args()

path= ""
out_path = f"{path}/{args.output_folder}/"

emb_dir = f"{path}/data/single_embeddings_v2/"
data_file = f'{path}/data/single_label_dataset.csv'
part_file = f"{path}/data/single_graphpart.csv"


uf.initialize_outer_metadata(path=out_path)
uf.enforce_reproducibility()
device = uf.get_gpu()
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


locations = ['Cellwall','Extracellular', 'Cytoplasmic', 'CYtoplasmicMembrane', 'OuterMembrane','Periplasmic']
epochs = int(args.epochs)

### hyperparameters###
hyper_combinations = uf.get_combinations()
outer_list = uf.outer_cross()

for idx, (lr, batch_size, drop_out) in enumerate(hyper_combinations):
    print(f"#####\n####\n -- Hyper combination {idx}/{len(hyper_combinations)} \t lr: {lr}, batch_size: {batch_size}, drop_out: {drop_out} #####\n####\n")

    ######outer validation###
    for test, subset in outer_list:
        inner_list = uf.inner_cross(subset)
        f1_macro_list = []

        ###inner validation###
        for validation, training in inner_list:
            training_loader, val_loader, weights = uf.preparing_data_loader(emb_dir, data_file, part_file, training, validation, batch_size, locations)
            
            model = uf.get_model(drop_out, device)
            optimizer, scheduler = uf.get_optimizers(model, lr)

            train_loss, val_loss = [],[]

            # take the model with the best epoch
            f1_macro_epoch, acc_epoch  = 0, 0

            #loss function and decide if use weights or not
            # cross_loss = nn.CrossEntropyLoss(weight = weights.to(device))
            cross_loss = nn.CrossEntropyLoss()
            
            for epoch in tqdm(range(epochs)):
            # for epoch in range(epochs):
                train_loss, val_loss = [],[]

                #training
                model.train()
                #adding weights to use focal loss
                model, running_train_loss = uf.training_loop(model, training_loader, cross_loss, optimizer, device)
                train_loss.append(running_train_loss/ len(training_loader))
                
                ##EVAL##
                model.eval()
                model, running_val_loss, all_val_predicts, all_val_trues = uf.validation_loop(model, val_loader, cross_loss, device)
                #convert predicts to int  
                all_val_predicts_int = uf.get_int_from_predicts(all_val_predicts)
                val_loss.append(running_val_loss / len(val_loader))

                if (epoch+1) % 5 == 0:
                    print(f'\t Epoch {epoch+1} \t train Loss: {running_train_loss / len(training_loader)} -- Val loss: {running_val_loss / len(val_loader)}')

                f1_macro = f1_score(all_val_trues, all_val_predicts_int, average="macro", zero_division=0)
                acc = accuracy_score(all_val_trues, all_val_predicts_int)

                if (f1_macro > f1_macro_epoch) & (acc > acc_epoch):
                    f1_macro_epoch = f1_macro
                    torch.save(model.state_dict(), f"{out_path}/models/model_{test}_{validation}")

                scheduler.step(running_val_loss / len(val_loader))

            print(f"\t\tINNER: test:{test}, validation:{validation}, training:{training} run -- at {uf.get_time()} \n")


            #save best f1_macro for that inner k validation
            f1_macro_list.append(f1_macro)

        #calculate average
        mean_f1 = sum(f1_macro_list) / len(f1_macro_list)

        #if that average is better than other f1 average with other hyperparameters we update the metadata
        uf.update_f1(out_path, test, mean_f1, lr, batch_size, drop_out)

        print(f"\t### \n\tOUTER: test:{test} done \n\t###")
        
# uf.get_losses_plot(train_loss, val_loss, out_path) #just checking if the model learns

print(" CROSS VALIDATION DONE ")
