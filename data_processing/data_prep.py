# data utils
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from operator import add
import os
import csv

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process data with a specified random seed.')
parser.add_argument('--seed', type=int, default=13, help='Random seed for reproducibility')
args = parser.parse_args()
seed = args.seed

# read in utils
from utility import *

# Define file paths
one_cvd_path = "data/model_data/one_cvd.csv"
two_cvd_path = "data/model_data/two_cvd.csv"
drugs_ordered_path = "data/model_data/drugs_ordered.csv"
one_cvd_drugs_ordered_path = "data/model_data/one_cvd_drugs_ordered.csv"
proteins_ordered_path = "data/model_data/proteins_ordered.csv"
drug_label_path = "data/model_data/embeddings/drug_label.npy"
dp_adj_path = "data/model_data/embeddings/dp_adj.npy"
mol_adj_path = "data/model_data/embeddings/mol_adj.npy"
cvd_drug_label_path = "data/model_data/embeddings/cvd_drug_label.npy"
cvd_dp_adj_path = "data/model_data/embeddings/cvd_dp_adj.npy"
cvd_mol_adj_path = "data/model_data/embeddings/cvd_mol_adj.npy"
se2combo_path = "data/model_data/embeddings/se2combo.npy"
ddi_adj_path = "data/model_data/embeddings/ddi_adj.npz"
edge_list_path = "data/model_data/edge_list.npz"
ddi_se_ordered_path = "data/model_data/ddi_se_ordered.csv"
cvd_ddi_se_ordered_path = "data/model_data/one_cvd_ddi_se_ordered.csv"
cvd_ddi_adj_path = "data/model_data/embeddings/one_cvd_ddi_adj.npz"
cvd_edge_list_path = "data/model_data/one_cvd_edge_list.npz"


# read in decagon data
combo2stitch, combo2se, se2name, net, node2idx, stitch2se, se2name_mono, stitch2proteins, se2class, se2name_class = read_decagon()

# read in smiles data
smiles = read_smiles()

# read in cvd df and create ordered lists
cvd_df = pd.read_csv("data/cvd_df.csv")

if not os.path.exists(one_cvd_path) or not os.path.exists(two_cvd_path):
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
    pd.DataFrame({"drug_pairs": one_cvd}).to_csv(one_cvd_path)
    pd.DataFrame({"drug_pairs": two_cvd}).to_csv(two_cvd_path)
else:
    print("- Loading existing cvd drug pairs")

# Read existing data if files already exist
one_cvd = pd.read_csv(one_cvd_path)
two_cvd = pd.read_csv(two_cvd_path)

# get counts and create ordered lists of all drugs and proteins

if not os.path.exists(drugs_ordered_path):
    drugs = set()
    for d in combo2stitch.keys():
        d1, d2 = d.split("_")
        drugs.add(d1)
        drugs.add(d2)  
    drugs = list(drugs)
    pd.DataFrame({"drugs": drugs}).to_csv(drugs_ordered_path, index=False)
else:
    print("- Loading existing drugs")
    drugs = list(pd.read_csv(drugs_ordered_path)["drugs"])
    
if not os.path.exists(one_cvd_drugs_ordered_path):
    one_cvd_drugs = set()
    for d in one_cvd["drug_pairs"]:
        d1, d2 = d.split("_")
        one_cvd_drugs.add(d1)
        one_cvd_drugs.add(d2)  
    one_cvd_drugs = list(one_cvd_drugs)
    pd.DataFrame({"drugs": one_cvd_drugs}).to_csv(one_cvd_drugs_ordered_path, index=False)
else:
    print("- Loading existing one cvd drugs")
    one_cvd_drugs = list(pd.read_csv(one_cvd_drugs_ordered_path)["drugs"])

if not os.path.exists(proteins_ordered_path):
    proteins = set()
    for p_set in stitch2proteins.values():
        proteins = proteins.union(p_set)
    proteins = [int(p) for p in list(proteins)]
    pd.DataFrame({"proteins": proteins}).to_csv(proteins_ordered_path, index=False)
else:
    print("- Loading existing proteins")
    proteins = list(pd.read_csv(proteins_ordered_path)["proteins"])

# get counts to help with matrix building
n_drugs = len(drugs)
n_proteins = len(proteins)
n_one_cvd_drugs = len(one_cvd_drugs)


# ------------------------------------ Construct Drug Features ----------------------------------------
print("Constructing Drug Features")

# construct mono se features
se_mono=[]
for k in se2name_mono:
    se_mono.append(k)
pd.DataFrame({"se_mono": se_mono}).to_csv("data/model_data/se_mono_ordered.csv", index=False)
    
if not os.path.exists(drug_label_path):
    drug_label = np.zeros((n_drugs, len(se_mono)))
    for key, value in stitch2se.items():
        j = drugs.index(key)
        for v in value:
            i = se_mono.index(v)
            drug_label[j, i] = 1
    np.save(drug_label_path, drug_label)
else:
    print("- Loading existing drug labels")

drug_label = np.load(drug_label_path)

if not os.path.exists(dp_adj_path):
    dp_adj = np.zeros((n_drugs, n_proteins))
    for i, drug_name in enumerate(drugs):
        for protein_name in stitch2proteins[drug_name]:
            k = proteins.index(int(protein_name))
            dp_adj[i, k] = 1
    np.save(dp_adj_path, dp_adj)
else:
    print("- Loading existing drug protein adj matrix")

dp_adj = np.load(dp_adj_path)

if not os.path.exists(mol_adj_path):
    molecule_embeddings = np.load('data/model_data/embeddings/drugname_smiles.npy')
    mol_adj = np.zeros((n_drugs, molecule_embeddings.shape[1]))
    smiles_drugs = list(smiles['drug_id'])
    for i, drug in enumerate(drugs):
        mol_index = smiles_drugs.index(drug)
        mol_adj[i] = molecule_embeddings[mol_index]
    np.save(mol_adj_path, mol_adj)
else:
    print("- Loading existing mol adj matrix")

mol_adj = np.load(mol_adj_path)




# ------------------------------------ Construct CVD Drug Features ----------------------------------------
print("Constructing CVD Drug Features")

if not os.path.exists(cvd_drug_label_path):
    cvd_drug_label = np.zeros((n_one_cvd_drugs, len(se_mono)))
    for key, value in stitch2se.items():
        if key in one_cvd_drugs:
            j = one_cvd_drugs.index(key)
            for v in value:
                i = se_mono.index(v)
                cvd_drug_label[j, i] = 1
    np.save(cvd_drug_label_path, cvd_drug_label)
else:
    print("- Loading existing cvd drug labels")

cvd_drug_label = np.load(cvd_drug_label_path)

if not os.path.exists(cvd_dp_adj_path):
    cvd_dp_adj = np.zeros((n_one_cvd_drugs, n_proteins))
    for i, drug_name in enumerate(one_cvd_drugs):
        for protein_name in stitch2proteins[drug_name]:
            k = proteins.index(int(protein_name))
            cvd_dp_adj[i, k] = 1
    np.save(cvd_dp_adj_path, cvd_dp_adj)
else:
    print("- Loading existing cvd drug protein adj matrix")

cvd_dp_adj = np.load(cvd_dp_adj_path)

if not os.path.exists(cvd_mol_adj_path):
    cvd_mol_adj = np.zeros((len(one_cvd_drugs), mol_adj.shape[1]))
    for i, drug_name in enumerate(one_cvd_drugs):
        mol_index = drugs.index(drug_name)
        cvd_mol_adj[i] = mol_adj[mol_index]
    np.save(cvd_mol_adj_path, cvd_mol_adj)
else:
    print("- Loading existing cvd mol adj matrix")

cvd_mol_adj = np.load(cvd_mol_adj_path)

# ------------------------------------ Construct Drug-Drug Interaction Matrices ----------------------------------------
print("Constructing DDI Matrices Features")

# Check if files exist before creating them
if not os.path.exists(se2combo_path) or not os.path.exists(ddi_se_ordered_path) or not os.path.exists(ddi_adj_path) or not os.path.exists(edge_list_path):

    se2combo = {}
    ddi_types = []
    
    for drug_pair, val in combo2se.items():
        for se in list(val):
            ddi_types.append(se)
            if se in se2combo.keys():
                se2combo[se].append(drug_pair)
            else:
                se2combo[se] = [drug_pair]

    # Save se2combo
    np.save(se2combo_path, se2combo)

    # Filter for side effects that occurred in at least 500 drug-drug pairs
    ddi_counter = Counter(ddi_types)
    ddi_counter_500 = Counter({k: c for k, c in ddi_counter.items() if c >= 500})

    n_ddi_types = len(ddi_counter_500)
    ddi_types = list(ddi_counter_500.keys())

    # Save ddi types to CSV
    pd.DataFrame({'ddi types': ddi_types}).to_csv(ddi_se_ordered_path, index=False)

    # Construct adjacency matrices and edge lists
    dd_adj_list = []
    edge_list = []

    for i, se in enumerate(ddi_types):
        mat = np.zeros((n_drugs, n_drugs))
        curr_edges = []

        for dp in se2combo[se]:
            d1, d2 = dp.split("_")
            drug1 = drugs.index(d1)
            drug2 = drugs.index(d2)
            mat[drug1, drug2] = 1
            mat[drug2, drug1] = 1
            curr_edges.append([dp, 1])

            # Add a negative sample
            added_neg_edge = False
            while not added_neg_edge:
                rand_d1, rand_d2 = np.random.randint(0, n_drugs), np.random.randint(0, n_drugs)
                if mat[rand_d1, rand_d2] == 0:
                    curr_edges.append([drugs[rand_d1] + "_" + drugs[rand_d2], 0])
                    added_neg_edge = True

        dd_adj_list.append(mat)
        edge_list.append(curr_edges)

    # Save adjacency matrices and edge lists
    np.savez(ddi_adj_path, *dd_adj_list)
    np.savez(edge_list_path, *edge_list)

else:
    # Load pre-existing files
    print("- Loading existing DDI matrices and edge lists")
    se2combo = np.load(se2combo_path, allow_pickle=True).item()
    ddi_types = list(pd.read_csv(ddi_se_ordered_path)['ddi types'])
    
     # Load adjacency matrices
    with np.load(ddi_adj_path, allow_pickle=True) as data:
        dd_adj_list = [data[key] for key in data.files]

    # Load edge lists
    with np.load(edge_list_path, allow_pickle=True) as data:
        edge_list = [data[key].tolist() for key in data.files]  # Convert to list of lists for consistency

# ------------------------------------ Construct Drug-Drug Interaction Matrices for Drug Pairs with at Least One CVD Drug ----------------------------------------
print("Constructing CVD DDI Matrices Features")

# Check if files exist before generating new data
if not os.path.exists(cvd_ddi_se_ordered_path) or not os.path.exists(cvd_ddi_adj_path) or not os.path.exists(cvd_edge_list_path):

    one_cvd = pd.read_csv("data/model_data/one_cvd.csv")
    one_cvd_set = []

    for key in one_cvd["drug_pairs"]:
        for se in list(combo2se[key]):
            one_cvd_set.append(se)

    # Filter for side effects that occurred in at least 500 drug-drug pairs
    one_cvd_ddi_counter = Counter(one_cvd_set)
    one_cvd_ddi_counter_500 = Counter({k: c for k, c in one_cvd_ddi_counter.items() if c >= 500})

    n_one_cvd_ddi_types = len(one_cvd_ddi_counter_500)
    one_cvd_ddi_types = list(one_cvd_ddi_counter_500.keys())

    # Save CVD DDI types to CSV
    pd.DataFrame({"cvd_ddi_se": one_cvd_ddi_types}).to_csv(cvd_ddi_se_ordered_path, index=False)

    # Format drug-drug adjacency matrix for each side effect
    one_cvd_dd_adj_list = []
    one_cvd_edge_list = []
    cvd_dps = set(one_cvd['drug_pairs'])

    for i, se in enumerate(one_cvd_ddi_types):
        mat = np.zeros((len(one_cvd_drugs), len(one_cvd_drugs)))
        curr_edges = []

        for dp in se2combo[se]:
            if dp in cvd_dps:
                d1, d2 = dp.split("_")
                drug1 = one_cvd_drugs.index(d1)
                drug2 = one_cvd_drugs.index(d2)
                mat[drug1, drug2] = 1
                mat[drug2, drug1] = 1
                curr_edges.append([dp, 1])

                # Add a negative sample
                added_neg_edge = False
                while not added_neg_edge:
                    rand_d1, rand_d2 = np.random.randint(0, len(one_cvd_drugs)), np.random.randint(0, len(one_cvd_drugs))
                    if mat[rand_d1, rand_d2] == 0:
                        curr_edges.append([one_cvd_drugs[rand_d1] + "_" + one_cvd_drugs[rand_d2], 0])
                        added_neg_edge = True

        one_cvd_dd_adj_list.append(mat)
        one_cvd_edge_list.append(curr_edges)

    # Save adjacency matrices and edge lists
    np.savez(cvd_ddi_adj_path, *one_cvd_dd_adj_list)
    np.savez(cvd_edge_list_path, *one_cvd_edge_list)

else:
    # Load existing files
    print("- Loading existing CVD DDI matrices and edge lists")
    one_cvd_ddi_types = list(pd.read_csv(cvd_ddi_se_ordered_path)["cvd_ddi_se"])
     # Load adjacency matrices
    with np.load(cvd_ddi_adj_path, allow_pickle=True) as data:
        one_cvd_dd_adj_list = [data[key] for key in data.files]

    # Load edge lists
    with np.load(cvd_edge_list_path, allow_pickle=True) as data:
        one_cvd_edge_list = [data[key].tolist() for key in data.files]  # Convert to list of lists for consistency


# ------------------------------------ Create Train Valid Test Split ----------------------------------------
print(f"Creating Train Test Validation Splits with Random Seed {seed}")

# For All Pairs
# for each SE, build a train, test, val split 
train_pairs, train_y = [], []
val_pairs, val_y = [], []
test_pairs, test_y = [], []

for i, se in enumerate(ddi_types):
    edges = edge_list[i].copy()
    np.random.seed(seed)
    np.random.shuffle(edges)
    a = len(edges) // 10
    val = [e[0] for e in edges[:a]]
    val_ys = [e[1] for e in edges[:a]]
    test = [e[0] for e in edges[a:a+a]]
    test_ys = [e[1] for e in edges[a:a+a]]
    train = [e[0] for e in edges[a+a:]]
    train_ys = [e[1] for e in edges[a+a:]]
    
    val_pairs.append(val)
    val_y.append(val_ys)
    test_pairs.append(test)
    test_y.append(test_ys)
    train_pairs.append(train)
    train_y.append(train_ys)

# save as csvs for replicability
datasets = [train_pairs, train_y, val_pairs, val_y, test_pairs, test_y]
names = ["train_pairs", "train_y", "val_pairs", "val_y", "test_pairs", "test_y"]

if not os.path.exists(f'data/model_data/TTS/{seed}/'):
    os.makedirs(f'data/model_data/TTS/{seed}/')

for data, name in zip(datasets, names):
    with open(f"data/model_data/TTS/{seed}/{name}.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(data)

# For CVD Pairs
train_pairs, train_y = [], []
val_pairs, val_y = [], []
test_pairs, test_y = [], []

for i, se in enumerate(one_cvd_ddi_types):
    edges = one_cvd_edge_list[i].copy()
    np.random.seed(seed)
    np.random.shuffle(edges)
    a = len(edges) // 10
    val = [e[0] for e in edges[:a]]
    val_ys = [e[1] for e in edges[:a]]
    test = [e[0] for e in edges[a:a+a]]
    test_ys = [e[1] for e in edges[a:a+a]]
    train = [e[0] for e in edges[a+a:]]
    train_ys = [e[1] for e in edges[a+a:]]
    
    val_pairs.append(val)
    val_y.append(val_ys)
    test_pairs.append(test)
    test_y.append(test_ys)
    train_pairs.append(train)
    train_y.append(train_ys)

# save as csvs for replicability
datasets = [train_pairs, train_y, val_pairs, val_y, test_pairs, test_y]
names = ["cvd/train_pairs", "cvd/train_y", "cvd/val_pairs", "cvd/val_y", "cvd/test_pairs", "cvd/test_y"]

if not os.path.exists(f'data/model_data/TTS/{seed}/cvd/'):
    os.makedirs(f'data/model_data/TTS/{seed}/cvd/')

for data, name in zip(datasets, names):
    with open(f"data/model_data/TTS/{seed}/{name}.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(data)