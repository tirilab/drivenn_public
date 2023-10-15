from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
import csv
import time

# read in utils
from utility import *
from training import *

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

# read in cvd dataframes
one_cvd = list(pd.read_csv("../data/one_cvd.csv")['drug_pairs'])
two_cvd = list(pd.read_csv("../data/two_cvd.csv")['drug_pairs'])


one_cvd_pairs = [dp.split("_") for dp in one_cvd]
two_cvd_pairs = [dp.split("_") for dp in two_cvd]

one_cvd_drugs, two_cvd_drugs, one_cvd_ddi_se, two_cvd_ddi_se  = read_cvd_lists()

n_one_cvd_drugs = len(one_cvd_drugs)
n_two_cvd_drugs = len(two_cvd_drugs)

print(f'num one cvd pairs: {len(one_cvd_pairs)} | num two cvd pairs: {len(two_cvd_pairs)} | num one cvd drugs: {n_one_cvd_drugs} | num two cvd drugs: {n_two_cvd_drugs}')
# ---------------------------------  get drug features --------------------------------- 

# mono se features
cvd_drug_label = np.zeros((n_one_cvd_drugs,len(se_mono)))
for key,value in stitch2se.items():
    if key in one_cvd_drugs:
        j = one_cvd_drugs.index(key)
    for v in value:
        i=se_mono.index(v)
        cvd_drug_label[j,i]=1
        
# drug protein features
cvd_dp_adj = np.zeros((n_one_cvd_drugs, n_proteins))
for i in range(n_one_cvd_drugs):
    drug_name = one_cvd_drugs[i]
    for protein_name in stitch2proteins[drug_name]:
        k = proteins.index(int(protein_name))
        cvd_dp_adj[i, k] = 1

# read in drug molecule features (embeddings from dgllife)
mol_adj  = np.load('../data/model_data/embeddings/mol_adj.npy')

cvd_mol_adj = np.zeros((n_one_cvd_drugs, mol_adj.shape[1]))

for i in range(n_one_cvd_drugs):
    drug_name = one_cvd_drugs[i]
    mol_index = drugs.index(drug_name)
    mol_row = mol_adj[mol_index]
    
    cvd_mol_adj[i] = np.array(mol_row)


cvd_drug_feat = np.concatenate((cvd_drug_label, cvd_dp_adj, cvd_mol_adj), axis=1)

print("full cvd drug feature shape: ", cvd_drug_feat.shape)

# --------------------------------- dd adj matrix --------------------------------- 

# format drug drug adjacency matrix for each side effect
one_cvd_dd_adj_list = []
two_cvd_dd_adj_list = []
one_cvd_set = set(one_cvd)
two_cvd_set = set(two_cvd)

for i in range(len(one_cvd_ddi_se)):
    if i%100 == 0:
        print(f'SE number: {i}')
    mat = np.zeros((n_one_cvd_drugs, n_one_cvd_drugs))
    curr_se = one_cvd_ddi_se[i]
    drug_pairs_for_se = se2combo[curr_se]
    for dp in drug_pairs_for_se:
        if dp in one_cvd_set:
            d1, d2 = dp.split("_")
            drug1 = one_cvd_drugs.index(d1)
            drug2 = one_cvd_drugs.index(d2)
            mat[drug1, drug2] = 1
            mat[drug2, drug1] = 1
    one_cvd_dd_adj_list.append(mat)

print('finished creating drug drug adj matrix for each side effect of one cvd')

for i in range(len(two_cvd_ddi_se)):
    if i%100 == 0:
        print(f'SE number: {i}')
    mat = np.zeros((n_two_cvd_drugs, n_two_cvd_drugs))
    curr_se = two_cvd_ddi_se[i]
    drug_pairs_for_se = se2combo[curr_se]
    for dp in drug_pairs_for_se:
        if dp in two_cvd_set:
            d1, d2 = dp.split("_")
            drug1 = two_cvd_drugs.index(d1)
            drug2 = two_cvd_drugs.index(d2)
            mat[drug1, drug2] = 1
            mat[drug2, drug1] = 1
    two_cvd_dd_adj_list.append(mat)

print('finished creating drug drug adj matrix for each side effect of two cvd')

# --------------------------------- get pos and neg edges --------------------------------- 

# get positive and negative edges
one_cvd_edges_all = []

for i in range(len(one_cvd_ddi_se)):
    if i%100 == 0:
        print(f'SE number: {i}')
    curr_se = one_cvd_ddi_se[i]
    curr_edges = se2combo[curr_se]
    curr_edges_set = set()
    for dp in curr_edges:
        if dp in one_cvd_set:
            curr_edges_set.add(dp)

    c = [pair.split("_") for pair in list(curr_edges_set)]
    
    # create an equal number of neg edges
    n = []
    while len(n) < len(c):
        d1, d2 = random.choice(one_cvd_drugs), random.choice(one_cvd_drugs)
        rand_pair, rand_pair_inv = f'{d1}_{d2}', f'{d2}_{d1}'
        if rand_pair not in curr_edges_set and rand_pair_inv not in curr_edges_set:
            n.append([d1, d2])
    
    # create edges_all
    all_edges = c + n
    np.random.shuffle(all_edges)
    one_cvd_edges_all.append(all_edges[:len(c)])


del c
del n

print('finished getting all edges for one cvd pairs')

# get positive and negative edges
two_cvd_edges_all = []

for i in range(len(two_cvd_ddi_se)):
    if i%100 == 0:
        print(f'SE number: {i}')
    curr_se = two_cvd_ddi_se[i]
    curr_edges = se2combo[curr_se]
    curr_edges_set = set()
    for dp in curr_edges:
        if dp in two_cvd_set:
            curr_edges_set.add(dp)

    c = [pair.split("_") for pair in list(curr_edges_set)]
    
    # create an equal number of neg edges
    n = []
    while len(n) < len(c):
        d1, d2 = random.choice(two_cvd_drugs), random.choice(two_cvd_drugs)
        rand_pair, rand_pair_inv = f'{d1}_{d2}', f'{d2}_{d1}'
        if rand_pair not in curr_edges_set and rand_pair_inv not in curr_edges_set:
            n.append([d1, d2])
    
    # create edges_all
    all_edges = c + n
    np.random.shuffle(all_edges)
    two_cvd_edges_all.append(all_edges[:len(c)])


del c
del n

print('finished getting all edges for two cvd pairs')

# --------------------------------- get input features --------------------------------- 
pca_val = 0.95

# create PCA model for mono se
mono_pca = PCA(pca_val)
mono_pca.fit(cvd_drug_label)
mono_se_pca_vector = mono_pca.transform(cvd_drug_label)

# create PCA model for dp 
dp_pca = PCA(pca_val)
dp_pca.fit(cvd_dp_adj)
dp_pca_vector = dp_pca.transform(cvd_dp_adj)

print('mono pca', mono_se_pca_vector.shape)
print('dp pca', dp_pca_vector.shape)
print('mol adj', cvd_mol_adj.shape)

# create full drug features
total_features = np.concatenate((mono_se_pca_vector, dp_pca_vector, cvd_mol_adj), axis=1)

print(f'Drug Features for PCA({pca_val}): {total_features.shape}')
del mono_se_pca_vector, dp_pca_vector

# --------------------------------- get data splits for each se and start cvd model training vs normal model training --------------------------------- 

run_type = 'cvd_training_final'
methods = 'one_cvd'


start_wandb(run_type, f'{methods}')
summary_for_model = []
for i in range(len(one_cvd_ddi_se)):
    train_x, train_y, val_x, val_y, test_x, test_y = single_data_split(total_features, i, one_cvd_edges_all, one_cvd_drugs, one_cvd_dd_adj_list)
    print(train_x.shape, val_x.shape, test_x.shape)
    start = time.time()
    scores = train_single_model(f'{run_type}/{methods}', i, one_cvd_ddi_se, train_x, train_y, val_x, val_y, test_x, test_y, se2name, model_type='DRIVEN')
    stop = time.time()
    duration = stop-start
    print('Total Train Time for SE: ', duration)
    scores.append(duration)
    summary_for_model.append(scores)
pd.DataFrame({"ddi": one_cvd_ddi_se, "scores": summary_for_model}).to_csv(f'../data/cvd_output/cvd_output_{methods}.csv')

wandb.finish()
    





