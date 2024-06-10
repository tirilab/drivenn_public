# data utils
import pandas as pd
import numpy as np
from collections import Counter
from operator import add
import random
from sklearn.decomposition import PCA

# read in utils
from utility import *

# read in decagon data
combo2stitch, combo2se, se2name, net, node2idx, stitch2se, se2name_mono, stitch2proteins, se2class, se2name_class = read_decagon()

# read in smiles data
smiles = read_smiles()

# get counts and create ordered lists of all drugs and proteins

# get list of all drugs
drugs = set()
for d in combo2stitch.keys():
    d1, d2 = d.split("_")
    drugs.add(d1)
    drugs.add(d2)  
drugs = list(drugs)
pd.DataFrame({"drugs": drugs}).to_csv("data/model_data/drugs_ordered.csv", index=False)

# get list of all one cvd drugs
one_cvd_drugs = set()
for d in pd.read_csv("data/model_data/one_cvd.csv")["drug_pairs"]:
    d1, d2 = d.split("_")
    one_cvd_drugs.add(d1)
    one_cvd_drugs.add(d2)  
one_cvd_drugs = list(one_cvd_drugs)
n_one_cvd_drugs = len(one_cvd_drugs)
pd.DataFrame({"drugs": one_cvd_drugs}).to_csv("data/model_data/one_cvd_drugs_ordered.csv", index=False)

# get list of all proteins
proteins = set()
for p_set in stitch2proteins.values():
    proteins = proteins.union(p_set)
proteins = [int(p) for p in list(proteins)]

# get counts to help with matrix building
n_drugs = len(drugs)
n_proteins = len(proteins)
pd.DataFrame({"proteins": proteins}).to_csv("data/model_data/proteins_ordered.csv", index=False)

# ------------------------------------ Construct Drug Features ----------------------------------------
print("Constructing Drug Features")

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
np.save('data/model_data/embeddings/drug_label.npy', drug_label)

# construct drug protein features
dp_adj = np.zeros((n_drugs, n_proteins))
for i in range(n_drugs):
    drug_name = drugs[i]
    for protein_name in stitch2proteins[drug_name]:
        k = proteins.index(int(protein_name))
        dp_adj[i, k] = 1
# save to npy file
np.save('data/model_data/embeddings/dp_adj.npy', dp_adj)

# read in drug molecule features (embeddings from dgllife)
molecule_embeddings  = np.load('data/model_data/embeddings/drugname_smiles.npy')

# format to match order of drugs
mol_adj = np.zeros((n_drugs, len(molecule_embeddings[0])))
smiles_drugs = list(smiles['drug_id'])

for i in range(n_drugs):
    drug = drugs[i]
    mol_index = smiles_drugs.index(drug)
    mol_adj[i] = molecule_embeddings[mol_index]

# save to npy file
np.save('data/model_data/embeddings/mol_adj.npy', mol_adj)

# ------------------------------------ Construct CVD Drug Features ----------------------------------------
print("Constructing CVD Drug Features")

# mono se features
cvd_drug_label = np.zeros((n_one_cvd_drugs, len(se_mono)))
for key,value in stitch2se.items():
    if key in one_cvd_drugs:
        j = one_cvd_drugs.index(key)
    for v in value:
        i=se_mono.index(v) 
        cvd_drug_label[j,i]=1

# save to npy file
np.save('data/model_data/embeddings/cvd_drug_label.npy', cvd_drug_label)
        
# drug protein features
cvd_dp_adj = np.zeros((n_one_cvd_drugs, n_proteins))
for i in range(n_one_cvd_drugs):
    drug_name = one_cvd_drugs[i]
    for protein_name in stitch2proteins[drug_name]:
        k = proteins.index(int(protein_name))
        cvd_dp_adj[i, k] = 1

# save to npy file
np.save('data/model_data/embeddings/cvd_dp_adj.npy', cvd_dp_adj)

# read in drug molecule features (embeddings from dgllife)
mol_adj  = np.load('data/model_data/embeddings/mol_adj.npy')

cvd_mol_adj = np.zeros((n_one_cvd_drugs, mol_adj.shape[1]))

for i in range(n_one_cvd_drugs):
    drug_name = one_cvd_drugs[i]
    mol_index = drugs.index(drug_name)
    mol_row = mol_adj[mol_index]
    
    cvd_mol_adj[i] = np.array(mol_row)

# save to npy file
np.save('data/model_data/embeddings/cvd_mol_adj.npy', cvd_mol_adj)

# ------------------------------------ Construct Drug-Drug Interaction Matrices ----------------------------------------
print("Constructing DDI Matrices Features")

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

# save to npy file
np.save('data/model_data/embeddings/se2combo.npy', se2combo)
        
# only use side effects that occurred in at least 500 drug-drug pairs 
ddi_counter = Counter(ddi_types)
ddi_counter_500 = Counter({k: c for k, c in ddi_counter.items() if c >= 500})

n_ddi_types = len(ddi_counter_500)
ddi_types = list(ddi_counter_500.keys())
pd.DataFrame({'ddi types': ddi_types}).to_csv("data/model_data/ddi_se_ordered.csv", index=False)

# format drug drug adjacency matrix for each side effect
dd_adj_list = []

for i in range(n_ddi_types):
    mat = np.zeros((n_drugs, n_drugs))
    curr_se = ddi_types[i]
    drug_pairs_for_se = se2combo[curr_se]
    dp_set = set(drug_pairs_for_se)
    for dp in dp_set:
        d1, d2 = dp.split("_")
        drug1 = drugs.index(d1)
        drug2 = drugs.index(d2)
        mat[drug1, drug2] = 1
        mat[drug2, drug1] = 1
    dd_adj_list.append(mat)

# save to npy file
np.savez('data/model_data/embeddings/ddi_adj', *dd_adj_list)

# ------------------------------------ Construct Drug-Drug Interaction Matrices for Drug Pairs with at Least One CVD Drug ----------------------------------------
print("Constructing CVD DDI Matrices Features")

# read in cvd df and create ordered lists
cvd_df = pd.read_csv("data/cvd_df.csv")

one_cvd = []
two_cvd = []

for key in list(combo2se.keys()):
    drugs = key.split("_")
    if len(drugs) > 1:
        if drugs[0] in set(cvd_df.total_drugs):
            one_cvd.append(key)
            if drugs[1] in set(cvd_df.total_drugs):
                two_cvd.append(key)
        elif drugs[1] in set(cvd_df.total_drugs):
            one_cvd.append(key)

pd.DataFrame({"drug_pairs": one_cvd}).to_csv("data/model_data/one_cvd.csv")
pd.DataFrame({"drug_pairs": two_cvd}).to_csv("data/model_data/two_cvd.csv")

one_cvd = pd.read_csv("data/model_data/one_cvd.csv")
one_cvd_set = []

for key in one_cvd["drug_pairs"]:
    for se in list(combo2se[key]):
        one_cvd_set.append(se)

# only use side effects that occurred in at least 500 drug-drug pairs 
one_cvd_ddi_counter = Counter(one_cvd_set)
one_cvd_ddi_counter_500 = Counter({k: c for k, c in one_cvd_ddi_counter.items() if c >= 500})

one_cvd_ddi_list = list(one_cvd_set)

n_one_cvd_ddi_types = len(one_cvd_ddi_counter_500)
one_cvd_ddi_types = list(one_cvd_ddi_counter_500.keys())

pd.DataFrame({"cvd_ddi_se": one_cvd_ddi_types}).to_csv("data/model_data/one_cvd_ddi_se_ordered.csv", index=None)

# format drug drug adjacency matrix for each side effect
one_cvd_dd_adj_list = []
cvd_dps = set(one_cvd['drug_pairs'])

for i in range(n_one_cvd_ddi_types):
    mat = np.zeros((n_one_cvd_drugs, n_one_cvd_drugs))
    curr_se = one_cvd_ddi_types[i]
    drug_pairs_for_se = se2combo[curr_se]
    dp_set = set(drug_pairs_for_se)
    for dp in dp_set:
        if dp in cvd_dps:
            d1, d2 = dp.split("_")
            drug1 = one_cvd_drugs.index(d1)
            drug2 = one_cvd_drugs.index(d2)
            mat[drug1, drug2] = 1
            mat[drug2, drug1] = 1
    one_cvd_dd_adj_list.append(mat)

#save to npy file
np.savez('data/model_data/embeddings/one_cvd_ddi_adj', *one_cvd_dd_adj_list)


# ------------------------------------ Get Edges (All and CVD) ----------------------------------------
print("Constructing Edges")
# np.random.seed(13)
# random.seed(42)

# get positive and negative edges
edges_all = []

for i in range(len(ddi_types)):
    curr_se = ddi_types[i]
    curr_edges = se2combo[curr_se]
    curr_edges_set = []
    for dp in curr_edges:
        curr_edges_set.append(dp)

    c = [pair.split("_") for pair in list(set(curr_edges_set))]
    
    # create an equal number of neg edges
    n = []
    while len(n) < len(c):
        d1, d2 = random.choice(drugs), random.choice(drugs)
        rand_pair, rand_pair_inv = f'{d1}_{d2}', f'{d2}_{d1}'
        if rand_pair not in set(curr_edges_set) and rand_pair_inv not in set(curr_edges_set):
            n.append([d1, d2])
    
    # create edges_all
    all_edges = c + n
    
    np.random.shuffle(all_edges)
    edges_all.append(all_edges[:len(c)])

del c, n

# save edges_all
np.savez('data/model_data/edges_all', *edges_all)

print("Constructing CVD Edges")

# get cvd positive and negative edges
one_cvd_edges_all = []

for i in range(len(one_cvd_ddi_types)):
    curr_se = one_cvd_ddi_types[i]
    curr_edges = se2combo[curr_se]
    curr_edges_set = []
    ce_set = set(one_cvd['drug_pairs'])
    for dp in curr_edges:
        if dp in ce_set:
            curr_edges_set.append(dp)

    c = [pair.split("_") for pair in list(set(curr_edges_set))]
    
    # create an equal number of neg edges
    n = []
    while len(n) < len(c):
        d1, d2 = random.choice(one_cvd_drugs), random.choice(one_cvd_drugs)
        rand_pair, rand_pair_inv = f'{d1}_{d2}', f'{d2}_{d1}'
        if rand_pair not in ce_set and rand_pair_inv not in ce_set:
            n.append([d1, d2])
    
    # create edges_all
    all_edges = c + n
    np.random.shuffle(all_edges)
    one_cvd_edges_all.append(all_edges[:len(c)])

del c, n

# save one_cvd_edges_all
np.savez('data/model_data/one_cvd_edges_all', *one_cvd_edges_all)

# ------------------------------------ Create Train Valid Test Split ----------------------------------------
print("Creating Train Test Validation Splits")

def get_dp_for_single_se(k, edges_all):
    # train valid test split for drug pairs per side effect
    val=[]
    test=[]
    train=[]

    a = len(edges_all[k])//10
    val.append(edges_all[k][:a])
    test.append(edges_all[k][a:a+a])
    train.append(edges_all[k][a+a:])

    return train[0], test[0], val[0]

# create train valid test splits with drug pairs and save for later
train_dps_list = []
valid_dps_list = []
test_dps_list = []

for se in range(len(ddi_types)):
    train_dps, valid_dps, test_dps = get_dp_for_single_se(se, edges_all)
    train_dps_list.append(train_dps)
    valid_dps_list.append(valid_dps)
    test_dps_list.append(test_dps)

np.savez('data/model_data/TTS/train_dps', *train_dps_list)
np.savez('data/model_data/TTS/valid_dps', *valid_dps_list)
np.savez('data/model_data/TTS/test_dps', *test_dps_list)

# create train valid test splits with one cvd drug pairs and save for later
one_cvd_train_dps_list = []
one_cvd_valid_dps_list = []
one_cvd_test_dps_list = []

for se in range(len(one_cvd_ddi_types)):
    one_cvd_train_dps, one_cvd_valid_dps, one_cvd_test_dps = get_dp_for_single_se(se, one_cvd_edges_all)
    one_cvd_train_dps_list.append(one_cvd_train_dps)
    one_cvd_valid_dps_list.append(one_cvd_valid_dps)
    one_cvd_test_dps_list.append(one_cvd_test_dps)

np.savez('data/model_data/TTS/one_cvd_train_dps', *one_cvd_train_dps_list)
np.savez('data/model_data/TTS/one_cvd_valid_dps', *one_cvd_valid_dps_list)
np.savez('data/model_data/TTS/one_cvd_test_dps', *one_cvd_test_dps_list)

