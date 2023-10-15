# data utils
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
import random
from operator import add
import scipy.sparse as sp
from sklearn import metrics

# model utils
from keras.layers.core import Dropout, Activation
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.models import load_model

# read in utils
from utility import *

# read in decagon data
combo2stitch, combo2se, se2name = load_combo_se()
net, node2idx = load_ppi()
stitch2se, se2name_mono = load_mono_se()
stitch2proteins = load_targets()
se2class, se2name_class = load_categories()
se2name.update(se2name_mono)
se2name.update(se2name_class)

# read in smiles data
smiles = pd.read_csv("data/model_data/id_SMILE.txt", sep="\t", header=None)

# format smiles data to have same form as CID0*
smiles.iloc[:, 0] = [f'CID{"0"*(9-len(str(drug_id)))}{drug_id}' for drug_id in smiles.iloc[:,0]]
smiles = smiles.rename(columns={0:'drug_id', 1:'smiles'})

# get counts and create lists of all drugs and proteins

# get list of all drugs
drugs = set()
for d in combo2stitch.keys():
    d1, d2 = d.split("_")
    drugs.add(d1)
    drugs.add(d2)  
drugs = list(drugs)
pd.DataFrame({"drugs": drugs}).to_csv("data/model_data/drugs_ordered.csv", index=False)

# get list of all proteins
proteins = set()
for p_set in stitch2proteins.values():
    proteins = proteins.union(p_set)
proteins = list(proteins)

# get counts to help with matrix building
n_drugs = len(drugs)
n_proteins = len(proteins)
pd.DataFrame({"proteins": proteins}).to_csv("data/model_data/proteins_ordered.csv", index=False)

print(f'total drugs: {n_drugs} | total proteins: {n_proteins} | total mono se: {len(se_mono)} | total polypharmacy se: {len(ddi_se)}')

# ------------------------------------ Construct Drug Features ----------------------------------------

# construct mono se features
se_mono=[]
for k in se2name_mono:
    se_mono.append(k)
    
drug_label = np.zeros((n_drugs,len(se_mono)))
for key,value in stitch2se.items():
    j = drugs.index(key)
    for v in value:
        i=se_mono.index(v)
        drug_label[j,i]=1
pd.DataFrame({"se_mono": se_mono}).to_csv("data/model_data/se_mono_ordered.csv", index=False)

# save to npy file
np.save('../data/model_data/embeddings/drug_label.npy', drug_label)

# construct drug protein features
dp_adj = np.zeros((n_drugs, n_proteins))
for i in range(n_drugs):
    drug_name = drugs[i]
    for protein_name in stitch2proteins[drug_name]:
        k = proteins.index(protein_name)
        dp_adj[i, k] = 1
# save to npy file
np.save('../data/model_data/embeddings/dp_adj.npy', dp_adj)


# read in drug molecule features (embeddings from dgllife)
molecule_embeddings  = np.load('../data/model_data/embeddings/drugname_smiles.npy')

# format to match order of drugs
mol_adj = np.zeros((n_drugs, len(molecule_embeddings[0])))
smiles_drugs = list(smiles['drug_id'])

for i in range(n_drugs):
    drug = drugs[i]
    mol_index = smiles_drugs.index(drug)
    mol_adj[i] = molecule_embeddings[mol_index]

# save to npy file
np.save('../data/model_data/embeddings/mol_adj.npy', mol_adj)

# ------------------------------------ Construct Drug-Drug Interaction Matrices ----------------------------------------
se2combo = {}
ddi_types = []
for drug_pair, val in combo2se.items():
    for se in list(val):
        ddi_types.append(se)
        if se in se2combo.keys():
            current_pairs = se2combo[se]
            current_pairs.append(drug_pair)
            se2combo[se] = current_pairs
        else:
            se2combo[se] = [drug_pair]
        
# only use side effects that occurred in at least 500 drug-drug pairs 
ddi_counter = Counter(ddi_types)
ddi_counter_500 = Counter({k: c for k, c in ddi_counter.items() if c >= 500})

n_ddi_types = len(ddi_counter_500)
ddi_types = list(ddi_counter_500.keys())
pd.DataFrame({'ddi types': ddi_types}).to_csv("data/model_data/ddi_se_ordered.csv", index=False)

# format drug drug adjacency matrix for each side effect
dd_adj_list = []

for i in range(n_ddi_types):
    if i%100 == 0:
        print(f'On DDI matrix for SE number: {i}')
    mat = np.zeros((n_drugs, n_drugs))
    curr_se = ddi_types[i]
    drug_pairs_for_se = se2combo[curr_se]
    for dp in drug_pairs_for_se:
        d1, d2 = dp.split("_")
        drug1 = drugs.index(d1)
        drug2 = drugs.index(d2)
        mat[drug1, drug2] = 1
        mat[drug2, drug1] = 1
    dd_adj_list.append(mat)

# save to npy file
np.savez('../data/model_data/embeddings/ddi_adj.npy', *dd_adj_list)






