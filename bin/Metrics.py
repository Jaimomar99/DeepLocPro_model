import numpy as np
import single_useful_functions as uf
import pandas as pd
import os
import warnings
from collections import Counter
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, help="Indicate the folder within ProDeepLoc")
args = parser.parse_args()

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
out_path = f"{path}/{args.output_folder}/"

predicts = [pd.read_csv(f"{out_path}final_5_dfs/{f}", index_col=0) for f in os.listdir(f"{out_path}/final_5_dfs")]
predicts = pd.concat(predicts, axis=0)
#Get the true labels
trues = [pd.read_csv(f"{out_path}trues_labels/{f}", index_col=0) for f in os.listdir(f"{out_path}trues_labels/")]
trues = pd.concat(trues,axis=0)
#kingdoms
kingdoms = [pd.read_csv(f"{out_path}kingdoms/{f}", index_col=0) for f in os.listdir(f"{out_path}kingdoms/")]
kingdoms = pd.concat(kingdoms,axis=0)
archaea = kingdoms[kingdoms.kingdom == 0].index
gram_pos = kingdoms[kingdoms.kingdom == 1].index
gram_neg = kingdoms[kingdoms.kingdom == 2].index

#take trues and predicts
archaea_trues= trues.loc[archaea]
archaea_predicts= predicts.loc[archaea]
gram_pos_trues= trues.loc[gram_pos]
gram_pos_predicts= predicts.loc[gram_pos]
gram_neg_trues= trues.loc[gram_neg]
gram_neg_predicts= predicts.loc[gram_neg]

#convert to int for calculating metrics
predicts_int = uf.get_int_from_predicts(np.array(predicts))
archaea_predicts_int = uf.get_int_from_predicts(np.array(archaea_predicts))
gram_pos_predicts_int = uf.get_int_from_predicts(np.array(gram_pos_predicts))
gram_neg_predicts_int = uf.get_int_from_predicts(np.array(gram_neg_predicts))

#convert periplasm and outer membrane to extracellular
archaea_predicts_int = uf.modify_arch_and_positive(archaea_predicts_int)
gram_pos_predicts_int = uf.modify_arch_and_positive(gram_pos_predicts_int)

#concatenate new arrays to calcualte overall
predicts_int = np.concatenate((archaea_predicts_int, gram_pos_predicts_int, gram_neg_predicts_int), axis=0 )
trues_int = np.concatenate((archaea_trues, gram_pos_trues, gram_neg_trues), axis=0)


#ProDeepLoc Performance
overall_metrics = uf.get_metrics(predicts_int, trues_int, "overall")
archaea_metrics = uf.get_metrics(archaea_predicts_int, archaea_trues.values, "archaea")
gram_pos_metrics = uf.get_metrics(gram_pos_predicts_int, gram_pos_trues.values, "gram_pos")
gram_neg_metrics = uf.get_metrics(gram_neg_predicts_int, gram_neg_trues.values, "gram_neg")

final_performance = pd.concat([overall_metrics, archaea_metrics, gram_pos_metrics,  gram_neg_metrics], axis=1)
final_performance.to_csv(f"{out_path}/overall_performance.csv", header=True, index=True )

print("ProDeepLoc performance calculated")

#Confusion matrix
fig, axs = plt.subplots(2, 2, figsize=(18, 16))

locations = ['Cellwall', 'Extracellular', 'Cytoplasmic', 'Cytoplasmic\nMembrane', 'Outer\nMembrane', 'Periplasmic']
archaea_locations = ['Cellwall', 'Extracellular', 'Cytoplasmic', 'Cytoplasmic\nMembrane']

# generate each plot and assign them to the corresponding subplot
uf.plot_conf_matrix(predicts_int, trues_int, locations, "All organisms", axs[0, 0], xlabel="")
uf.plot_conf_matrix(archaea_predicts_int, archaea_trues.values, archaea_locations, "Archaea", axs[0, 1], xlabel="", ylabel="")
uf.plot_conf_matrix(gram_pos_predicts_int, gram_pos_trues.values, archaea_locations, "Gram positive", axs[1, 0])
uf.plot_conf_matrix(gram_neg_predicts_int, gram_neg_trues.values, locations, "Gram Negative", axs[1, 1], ylabel="")

# Adjust the spacing between subplots 
plt.subplots_adjust(hspace=0.25, wspace=0.15)

plt.savefig(f"{out_path}/confusion_matrix.png")

print("Confusion matrix created \n DONE")