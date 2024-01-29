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

path= ""
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


###Benchmarking with psort
print("prodeepLoc performance after 2010 psort")

date = pd.read_csv(f"{path}/data/date_df.csv", index_col=0)
date['date'] = pd.to_datetime(date['date'])
filter_date = pd.to_datetime("2010-01-01")
date_2010 = date[date["date"] >= filter_date]

predicts_2010 = predicts[predicts.index.isin(date_2010.index)]
trues_2010 = trues[trues.index.isin(date_2010.index)]
archaea_trues_2010= trues_2010[trues_2010.index.isin(archaea)]
archaea_predicts_2010= predicts_2010[predicts_2010.index.isin(archaea)]
gram_pos_trues_2010= trues_2010[trues_2010.index.isin(gram_pos)]
gram_pos_predicts_2010= predicts_2010[predicts_2010.index.isin(gram_pos)]
gram_neg_trues_2010= trues_2010[trues_2010.index.isin(gram_neg)]
gram_neg_predicts_2010= predicts_2010[predicts_2010.index.isin(gram_neg)]


predicts_int = uf.get_int_from_predicts(np.array(predicts_2010))
archaea_predicts_int_2010 = uf.get_int_from_predicts(np.array(archaea_predicts_2010))
gram_pos_predicts_int_2010 = uf.get_int_from_predicts(np.array(gram_pos_predicts_2010))
gram_neg_predicts_int_2010 = uf.get_int_from_predicts(np.array(gram_neg_predicts_2010))


archaea_predicts_int_2010 =  uf.modify_arch_and_positive(archaea_predicts_int_2010)
gram_pos_predicts_int_2010 = uf.modify_arch_and_positive(gram_pos_predicts_int_2010)

archaea_metrics_2010 = uf.get_metrics(archaea_predicts_int_2010, archaea_trues_2010.values, "archaea")
grampos_metrics_2010 = uf.get_metrics(gram_pos_predicts_int_2010, gram_pos_trues_2010.values, "gram_pos")
gramneg_metrics_2010 = uf.get_metrics(uf.get_int_from_predicts(np.array(gram_neg_predicts_2010)), gram_neg_trues_2010.values, "gram_neg")

final_performance = pd.concat([ archaea_metrics_2010, grampos_metrics_2010,  gramneg_metrics_2010], axis=1)
final_performance.to_csv(f"{out_path}/prodeeploc_2010_performance.csv", header=True, index=True )

###Psort no blast performance after 2010
print("Psort performance with no blast after 2010 report")
psort_path = f"{path}/psort_predictions/no_blast/"

archaea_no_blast  = uf.get_df_psort(f"{psort_path}archaea_predictions.txt")
gram_pos_no_blast = uf.get_df_psort(f"{psort_path}gram_pos_predictions.txt")
gram_neg_no_blast = uf.get_df_psort(f"{psort_path}gram_neg_predictions.txt")

archaea_no_blast_2010 = archaea_no_blast[archaea_no_blast.index.isin(date_2010.index)]
gram_pos_no_blast_2010 = gram_pos_no_blast[gram_pos_no_blast.index.isin(date_2010.index)]
gram_neg_no_blast_2010 = gram_neg_no_blast[gram_neg_no_blast.index.isin(date_2010.index)]

#archaea
trues_archaea_no_blast_2010, predicts_archaea_no_blast_2010 = uf.get_trues_predicts_psort(archaea_no_blast_2010)
archaea_psort = uf.get_metrics(trues_archaea_no_blast_2010, predicts_archaea_no_blast_2010, "archaea")

#gram_pos
trues_gram_pos_no_blast_2010, predicts_gram_pos_no_blast_2010 = uf.get_trues_predicts_psort(gram_pos_no_blast_2010)
gram_pos_psort = uf.get_metrics(trues_gram_pos_no_blast_2010, predicts_gram_pos_no_blast_2010, "gram_pos")

#gram_neg
trues_gram_neg_no_blast_2010, predicts_gram_neg_no_blast_2010 = uf.get_trues_predicts_psort(gram_neg_no_blast_2010)
gram_neg_psort = uf.get_metrics(trues_gram_neg_no_blast_2010, predicts_gram_neg_no_blast_2010, "gram_neg")

psort_2010_no_blast_performance = pd.concat([ archaea_psort, gram_pos_psort,  gram_neg_psort], axis=1)
psort_2010_no_blast_performance.to_csv(f"{out_path}/psort_2010_no_blast_performance.csv", header=True, index=True )