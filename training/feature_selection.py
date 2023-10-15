# data utils
import numpy as np
from sklearn.decomposition import PCA

import umap
from sklearn.preprocessing import StandardScaler

# read in utils
from utility import *
from training import *

# logging utils
import wandb
import time

# ---------------------------------  read in data --------------------------------- 
# read in decagon data
combo2stitch, combo2se, se2name = load_combo_se()
se2combo = np.load('../data/model_data/se2combo.npy',allow_pickle='TRUE').item()
stitch2se, se2name_mono = load_mono_se()
stitch2proteins = load_targets()
se2name.update(se2name_mono)

# get ordered lists
drugs, proteins, se_mono, ddi_se = read_ordered_lists()

# get counts to help with matrix building
n_drugs = len(drugs)
n_proteins = len(proteins)

print(f'total drugs: {n_drugs} | total proteins: {n_proteins} | total mono se: {len(se_mono)} | total polypharmacy se: {len(ddi_se)}')

# ---------------------------------  get drug features --------------------------------- 

# mono se features
drug_label = np.zeros((n_drugs,len(se_mono)))
for key,value in stitch2se.items():
    j = drugs.index(key)
    for v in value:
        i=se_mono.index(v)
        drug_label[j,i]=1
        
# drug protein features
dp_adj = np.zeros((n_drugs, n_proteins))
for i in range(n_drugs):
    drug_name = drugs[i]
    for protein_name in stitch2proteins[drug_name]:
        k = proteins.index(int(protein_name))
        dp_adj[i, k] = 1

# read in drug molecule features (embeddings from dgllife)
mol_adj  = np.load('../data/model_data/embeddings/mol_adj.npy')

print('done loading drug features')

# --------------------------------- dd adj matrix --------------------------------- 

# format drug drug adjacency matrix for each side effect
dd_adj_list = []

for i in range(len(ddi_se)):
    if i%100 == 0:
        print(f'SE number: {i}')
    mat = np.zeros((n_drugs, n_drugs))
    curr_se = ddi_se[i]
    drug_pairs_for_se = se2combo[curr_se]
    for dp in drug_pairs_for_se:
        d1, d2 = dp.split("_")
        drug1 = drugs.index(d1)
        drug2 = drugs.index(d2)
        mat[drug1, drug2] = 1
        mat[drug2, drug1] = 1
    dd_adj_list.append(mat)
print('finished creating drug drug adj matrix for each side effect')

# --------------------------------- get pos and neg edges --------------------------------- 

# get positive and negative edges
edges_all = []

for i in range(len(ddi_se)):
    if i%100 == 0:
        print(f'SE number: {i}')
    curr_se = ddi_se[i]
    curr_edges = se2combo[curr_se]
    curr_edges_set = set(curr_edges)
    c = [pair.split("_") for pair in curr_edges]
    
    # create an equal number of neg edges
    n = []
    while len(n) < len(curr_edges):
        d1, d2 = random.choice(drugs), random.choice(drugs)
        rand_pair, rand_pair_inv = f'{d1}_{d2}', f'{d2}_{d1}'
        if rand_pair not in curr_edges_set and rand_pair_inv not in curr_edges_set:
            n.append([d1, d2])
    
    # create edges_all
    all_edges = c + n
    np.random.shuffle(all_edges)
    edges_all.append(all_edges[:len(c)])
del c
del n
print('finished getting all edges')

# --------------------------------- start training --------------------------------- 

pca_vals = [0.85, 0.90, 0.95, 0.99]
models = ["pca", "UMAP"]
# update batch norm and if you want to use molecular embeddings here #
bn = False
mol = False
run_type = f'feature_selection_{bn}_bn_{mol}_mol'

print(f'staring training with {mol} molecular embeddings and {bn} batchnorm')

# to create summary df
summaries = {}

for method in models:
    if method == "pca":
        for pca_val in pca_vals:

            # create PCA model for mono se
            mono_pca = PCA(pca_val)
            mono_pca.fit(drug_label)
            mono_se_pca_vector = mono_pca.transform(drug_label)
            
            # create PCA model for dp 
            dp_pca = PCA(pca_val)
            dp_pca.fit(dp_adj)
            dp_pca_vector = dp_pca.transform(dp_adj)
            
            # create full drug features
            if mol:
                total_features = np.concatenate((mono_se_pca_vector, dp_pca_vector, mol_adj), axis=1)
            else:
                total_features = np.concatenate((mono_se_pca_vector, dp_pca_vector), axis=1)
            print(f'Drug Features for PCA({pca_val}): {total_features.shape}')
            del mono_se_pca_vector, dp_pca_vector
            
            summary_for_model = []
            
            # start a new wandb run to track this script
            start_wandb(run_type, f'{method}_{pca_val}')
            
            for i in range(len(ddi_se)):
                train_x, train_y, val_x, val_y, test_x, test_y = single_data_split(total_features, i, edges_all, drugs, dd_adj_list)
                print(train_x.shape, val_x.shape, test_x.shape)
                start = time.time()
                scores = train_single_model(f'{run_type}/{method}_{pca_val}', i, ddi_se, train_x, train_y, val_x, val_y, test_x, test_y, se2name, bn)
                stop = time.time()
                duration = stop-start
                print('Total Train Time for SE: ', duration)
                scores.append(duration)
                summary_for_model.append(scores)
            pd.DataFrame({"ddi": ddi_se, "scores": summary_for_model}).to_csv(f'../data/fs_output_{bn}_bn_{mol}_mol_pca_{pca_val}.csv')            
            summaries[f'pca_{pca_val}'] = summary_for_model
            wandb.finish()
    elif method == "UMAP":
        if mol:
            drug_feat = np.concatenate((drug_label, dp_adj, mol_adj), axis=1)
        else:
            drug_feat = np.concatenate((drug_label, dp_adj), axis=1)

        print("full drug feature shape: ", drug_feat.shape)

        
        # scale data
        scaled_drug_feat = StandardScaler().fit_transform(drug_feat)
        
        # use UMAP to reduce dimensionality
        reducer = umap.UMAP()
        umap_feat = reducer.fit_transform(scaled_drug_feat)  
        print(f'Drug Features for UMAP: {umap_feat.shape}')
        del scaled_drug_feat
        
        summary_for_model = []

        # start a new wandb run to track this script
        start_wandb(run_type, f'{method}')
            
        for i in range(len(ddi_se)):
            train_x, train_y, val_x, val_y, test_x, test_y = single_data_split(umap_feat, i, edges_all, drugs, dd_adj_list)
            print(train_x.shape, val_x.shape, test_x.shape)
            start = time.time()
            scores = train_single_model(f'feature_selection/umap', i, ddi_se, train_x, train_y, val_x, val_y, test_x, test_y, se2name, bn)
            stop = time.time()
            duration = stop-start
            print('Total Train Time for SE: ', duration)
            summary_for_model.append(scores)
        pd.DataFrame({"ddi": ddi_se, "scores": summary_for_model}).to_csv(f'../data/fs_output_{bn}_bn_{mol}_mol_umap.csv')
        summaries['umap'] = summary_for_model
        wandb.finish()

# # create model summaried df and save to csv

# se, roc_score, aupr_score, precision, recall, f_score, acc, mcc, duration = [], [], [], [], [], [], [], [], []
# mod = []
# method = list(summaries.keys())

# for m in method:
# 	all_summaries_for_method = summaries[m]
# 	for i in len(all_summaries_for_method):
# 		curr_row = all_summaries_for_method[i]
# 		mod.append(m)
# 		se.append(curr_row[0])
# 		roc_score.append(curr_row[1])
# 		aupr_score.append(curr_row[2]) 
# 		precision.append(curr_row[3])
# 		recall.append(curr_row[4])
# 		f_score.append(curr_row[5])
# 		acc.append(curr_row[6])
# 		mcc.append(curr_row[7])
# 		duration.append(curr_row[8])


# pd.DataFrame({"model": mod, "se": se, "roc_score": roc_score, "aupr_score": aupr_score, "precision": precision, "recall": recall, "f_score": f_score, "acc": acc, "mcc": mcc, "train_time": duration}).to_csv("data/feature_selection_summary_df.csv")




















