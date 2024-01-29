import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from single_model import DeepLocModel
import itertools
import pandas as pd
from compute_dataset import PrecomputedCSVDataset
import torch.nn as nn
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.metrics import jaccard_score, f1_score, matthews_corrcoef, accuracy_score, confusion_matrix


def INT_TO_KING(key):
    dict_to_king = {0:"ARCHAEA",
                    1:"POSITIVE",
                    2:"NEGATIVE"}
    return dict_to_king[key]


def LABEL_TO_INT(key):
    dic_to_int = {0:"Cellwall",
                1:"Extracellular",
                2:"Cytoplasmic",
                3:"CYtoplasmicMembrane",
                4:"OuterMembrane",
                5:"Periplasmic"}
    return dic_to_int[key]


def INT_TO_LABEL(key):
    dic_to_label = {
        "Cellwall": 0,
        "Extracellular": 1,
        "Cytoplasmic": 2,
        "CYtoplasmicMembrane": 3,
        "OuterMembrane": 4,
        "Periplasmic": 5
    }
    if key in dic_to_label:
        return dic_to_label[key]
    else:
        return "Unknown"


def get_combinations(lr=[0.001], batch_size=[16,32, 64], drop = [0.1, 0.2, 0.4]):
    ''' all possible combination of the parameters'''
    return [x for x in itertools.product(lr, batch_size, drop)]


def get_gpu():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def outer_cross():
    outer_list = []
    for test in range(5):
        others = [num for num in range(5) if num != test]
        outer_list.append((test,others))
    return outer_list


def inner_cross(outer_k_subset):
    '''create average at some point'''
    inner_list = []
    for validation in outer_k_subset:
        training = [num for num in outer_k_subset if num != validation]
        inner_list.append((validation, training))
    return inner_list


def initialize_outer_metadata(path=""):
    '''Function to compare three metrics and then take best one'''
    index = ["test" + str(n) for n in range(5)]
    row = [0, 0, 0 ,0]
    df = pd.DataFrame((row, row, row, row, row),columns=["f1_macro", "lr","bs", "drop"], index=index)
    print("Metadata file created")
    if len(path) == 0:
        print("please provide path")
    else:
        df.to_csv(f"{path}/metadata_single.csv", index=True)


def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently, no simple way to enforce determinism, as the order of parallel operations is not known.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_focal_loss(input, target, weights, device, gamma=1):
    bceloss = F.binary_cross_entropy_with_logits(input, target, pos_weight=weights.to(device), reduction="none")
    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(logpt)
    # compute the loss
    focal_loss = ( (1-pt) ** gamma ) * bceloss
    return focal_loss.mean()


def preparing_data_loader(emb_dir, data_file, part_file, training_subset, validation_subset, batch_size, locations):
    training_dataset = PrecomputedCSVDataset(embeddings_dir=emb_dir, data_file= data_file, partitioning_file=part_file, partitions=training_subset)
    validation_dataset = PrecomputedCSVDataset(embeddings_dir=emb_dir, data_file= data_file, partitioning_file=part_file, partitions=[validation_subset])
    classes = len(locations)
    total = len(training_dataset.data)
    weights = torch.Tensor([total/(classes*sum(training_dataset.data[location])) for location in locations])

    # sampler_training = WeightedRandomSampler(weights, len(weights)) #model was overfitting with this
    # training_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, collate_fn=training_dataset.collate_fn, sampler=sampler_training)
    training_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, collate_fn=training_dataset.collate_fn)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, collate_fn=validation_dataset.collate_fn)
    return training_loader, validation_loader, weights


def get_model(drop_out, device):
    model = DeepLocModel(embedding_dim=1280, num_classes=6, drop=drop_out)
    return model.to(device)


def get_optimizers(model, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
    return optimizer, scheduler


def training_loop(model, training_loader, loss_function, optimizer, device):
    running_train_loss = 0
    for i, batch in enumerate(training_loader):
        embeddings, mask, train_labels, kingdoms = batch
        embeddings, mask, train_labels, kingdoms = embeddings.to(device), mask.to(device), train_labels.to(device), kingdoms.to(device)     
        #prediction = forward pass
        train_predicts = model(embeddings,mask)
        #loss
        l = loss_function(train_predicts, train_labels.float())
        #calculate gradients,backward
        l.backward()
        #update weights
        optimizer.step()
        #zero gradients after updating
        optimizer.zero_grad()
        #add 
        running_train_loss += l.item() * mask.size(0)
    return model, running_train_loss


def validation_loop(model, val_loader, loss_function, device):
    running_val_loss = 0
    all_val_predicts, all_val_trues, all_val_kings = [], [],[]
    softmax = nn.Softmax(dim=1) #change to softmax, which dimension is my prob
    for i, batch in enumerate(val_loader):
        embeddings, mask, val_labels, kingdoms = batch
        embeddings, mask, val_labels, kingdoms = embeddings.to(device), mask.to(device), val_labels.to(device), kingdoms.to(device)

        #prediction = forward pass
        val_predicts = model(embeddings,mask)

        #loss_function
        val_l = loss_function(val_predicts, val_labels.float())

        all_val_predicts.append(softmax(val_predicts).cpu().detach().numpy())
        all_val_trues.append(val_labels.cpu().detach().numpy())
        # all_val_kings.append(kingdoms.cpu().detach().numpy())
        running_val_loss += val_l.item() * mask.size(0)
    #concatenate 
    all_val_predicts = np.concatenate(all_val_predicts, axis=0)
    all_val_trues = np.concatenate(all_val_trues, axis =0)
    # all_val_kings = np.concatenate(all_val_kings, axis=0)
    return model, running_val_loss, all_val_predicts, all_val_trues


def test_loop(model, test_loader, device ):
        model.eval()
        softmax = nn.Softmax(dim=1) #change to softmax as well, check dimension
        all_test_predicts, all_test_trues, all_test_kings = [], [], []
        for i, batch in enumerate(test_loader):
                embeddings, mask, test_trues, kingdoms = batch
                embeddings, mask, test_trues, kingdoms = embeddings.to(device), mask.to(device), test_trues.to(device), kingdoms.to(device)
                #prediction = forward pass
                test_predicts = model(embeddings,mask)
                
                all_test_predicts.append(softmax(test_predicts).cpu().detach().numpy())
                all_test_trues.append(test_trues.cpu().detach().numpy())
                all_test_kings.append(kingdoms.cpu().detach().numpy())

        all_test_predicts = np.concatenate(all_test_predicts, axis=0)
        # all_test_int = get_int_from_predicts(all_test_predicts)
        all_test_trues = np.concatenate(all_test_trues, axis =0)
        all_test_kings = np.concatenate(all_test_kings, axis=0)
        return all_test_predicts, all_test_trues, all_test_kings


def get_int_from_predicts(predicts):
    return np.array([[(x == max(subarr)) * 1 for x in subarr] for subarr in predicts])


def get_losses_plot(train_loss, val_loss, path):
    fig,ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].plot(train_loss)
    ax[0].set_title("Training Loss")
    ax[1].plot(val_loss)
    ax[1].set_title("Validation Loss")
    plt.figtext(0.5, 0.01, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
    plt.savefig(f"{path}/train_val_single_loss_plot.png")


def update_f1(path, test, f1_macro, lr, bs, drop):
    '''Update the metadata in the outer file, mean of F1'''
    df = pd.read_csv(f"{path}/metadata_single.csv", index_col=0)
    new_row = [f1_macro, lr, bs, drop]
    test_name = "test" + str(test)

    if f1_macro > df.loc[test_name, "f1_macro"]:
        df.loc[test_name, :] = new_row
    df.to_csv(f"{path}/metadata_single.csv", index=True)


def get_time():
    formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_date)




# def get_best_models(model, val_predicts_int, val_trues, val_loss_epoch, lr, bs, drop, path, extra_inf):
#     ''' Compare three different metrics to save the best model, and a metadata file'''
#     from sklearn.metrics import  f1_score

#     df = pd.read_csv(f"{path}/metadata_single{extra_inf}.csv", index_col=0)
#     acc = np.all((val_predicts_int == val_trues), axis=1).mean()
#     f1_macro = f1_score(val_trues, val_predicts_int, average="macro", zero_division=0)

#     new_row = [acc, val_loss_epoch, f1_macro, lr, bs, drop]

#     if acc > df.loc["acc", "acc"]:
#         torch.save(model.state_dict(), f"{path}/models/acc{extra_inf}")
#         df.loc["acc", :] = new_row
#         # plot_conf_mat(val_predicts_int, val_trues, path,  f"acc{extra_inf}")

#     if val_loss_epoch < df.loc["val_loss_epoch", "val_loss_epoch"]:
#         torch.save(model.state_dict(), f"{path}/models/val_loss_epoch{extra_inf}")
#         df.loc["val_loss_epoch", :] = new_row
#         # plot_conf_mat(val_predicts_int, val_trues, path,  f"val_loss_epoch{extra_inf}")

#     if f1_macro > df.loc["f1_macro", "f1_macro"]:
#         torch.save(model.state_dict(), f"{path}/models/f1_macro{extra_inf}")
#         df.loc["f1_macro", :] = new_row
#         # plot_conf_mat(val_predicts_int, val_trues, path,  f"f1_macro{extra_inf}")
#     df.to_csv(f"{path}/metadata_single{extra_inf}.csv", index=True)


def metrics_dict_to_table(dic):
    df = pd.DataFrame.from_dict(dic, orient='index')
    df.columns = ["metrics"]
    # Separate the 'MCC' column into a new DataFrame
    mcc_df = pd.DataFrame.from_dict(df.loc['MCC'].item(), orient='index')
    mcc_df.columns = ['MCC']
    # Merge the two tables horizontally
    merged_df = pd.concat([df.drop('MCC'), mcc_df], axis=1)
    return merged_df


def get_metrics(predicts_int, trues, title="select"):
    # https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea
    """
    Calculate various metrics for evaluating the performance of a classification model.
    """ 
    metrics = {}

    # metrics["pred_num_labels"] = f"{predicts_int.sum(1).mean():.3f}"
    metrics["ACC"] = f"{accuracy_score(trues, predicts_int):.3f}"
    # metrics["Jaccard"] = f"{jaccard_score(trues, predicts_int, average='weighted', zero_division=0):.3f}"
    # metrics["MicroF1"] = f"{f1_score(trues, predicts_int, average='micro', zero_division=0):.3f}"
    metrics["MacroF1"] = f"{f1_score(trues, predicts_int, average='macro', zero_division=0):.3f}"
    metrics["MCC"] = {}
    for i in range(6):
        metrics["MCC"][LABEL_TO_INT(i)] = f"{matthews_corrcoef(trues[:,i], predicts_int[:,i]):.3f}"
    
    metrics = metrics_dict_to_table(metrics)
    metrics["MCC"].fillna(metrics["metrics"], inplace=True)
    metrics.drop("metrics", axis=1, inplace=True)
    metrics.rename(columns = {"MCC": title}, inplace=True)
    metrics = metrics.astype(float).round(2)
    return metrics


def format_prediction_from_fasta(localization):
    "get prediction from fasta file psort, if unknown 0"
    n = INT_TO_LABEL(localization)
    if n == "Unknown": return [0,0,0,0,0,0]
    stri = [0,0,0,0,0,0]
    stri[n] = 1
    return stri


def filter_unknown(df, path): 
    #not using this right now, since format_prediction_from_fasta function, we remove Unknown for all 0s
    unknown = df[df.prediction == "Unknown"]
    print(f"Number of unknowns: {unknown.shape[0]} out of {df.shape[0]} in {path} dataset")
    df_pred = df[df.prediction != "Unknown"]
    return df_pred, unknown


def get_df_psort(path, subset=None):
    df = pd.read_csv(path, sep = "\t", index_col=0)
    df.loc[df.Localization == "CytoplasmicMembrane", "Localization" ] = "CYtoplasmicMembrane"
    df["ID"] = [name[0] for name in df.index.str.split("|")]
    df['trues'] = df.index.str.extract(r'=(\d+)')[0].to_list()
    df["trues"] = df.trues.apply(lambda x: [int(digit) for digit in str(x)])
    df["trues"] = [np.array(sublist) for sublist in df.trues]
    df["prediction"] = df.Localization.apply(format_prediction_from_fasta)
    df["prediction"] = [np.array(sublist) for sublist in df.prediction]
    df.set_index('ID', inplace=True)
    if subset!= None:
        df = df[df.index.isin(subset)]
    # df, unknown= filter_unknown(df, path)
    return df


def get_trues_predicts_psort(df):
    trues = np.array([np.array(sublist) for sublist in df.trues])
    predicts = np.array([np.array(sublist) for sublist in df.prediction])
    return trues, predicts


def get_date_uniprot(uniprot_id:str) -> datetime:
    '''get datetime when it was first publicated'''
    # print(f"accesing {uniprot_id}")
    try:
        up_info = requests.get(f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.json').json()
        date = up_info["entryAudit"]["firstPublicDate"]
        date_object = datetime.strptime(date, '%Y-%m-%d')
        return date_object
    except: #some cases entry is obsolete
        return "Unavailable"


def plot_conf_matrix(predicts_int, trues, locations, title="", ax= "", xlabel="Predicted Labels", ylabel = "True Labels"):
    predicts_single = predicts_int.argmax(1)
    trues_single = trues.argmax(1)
    cm = confusion_matrix(trues_single, predicts_single)
    cm_percent = (cm.astype("float").T / cm.sum(axis=1)).T * 100

    # plt.figure(figsize=(7, 7))
    sns.heatmap(cm_percent, annot=False, fmt='.1f', cmap='Reds', cbar=True, linewidths=0.1, ax=ax)
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        n = format(cm[i, j], 'd')
        percentage = '{:.1f}%'.format(cm_percent[i, j])
        ax.text(j + 0.5, i + 0.5, f'{n}\n{percentage}', ha='center', va='center', color="white" if cm[i, j] > thresh else "black", fontsize=13)

    ax.set_xticks(ticks = np.arange(len(locations)) + 0.5, labels=locations, rotation=-45, fontsize=13)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_yticks(np.arange(len(locations)) + 0.5, locations, rotation=0, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel , fontsize=16)
    ax.set_title(title, fontsize=20)
    plt.tight_layout()


def modify_arch_and_positive(arr):
    ''' Convert periplasm(5) and outer membrane(4) to extracellular(1)'''
    max_indices = np.argmax(arr, axis=1)
    mask = (max_indices == 5) | (max_indices == 4)
    arr[mask, max_indices[mask]] = 0
    arr[mask, 1] = 1
    return arr