import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import time

from training import *
from utility import *

# read in saved data
combo2stitch, combo2se, se2name, net, node2idx, stitch2se, se2name_mono, stitch2proteins, se2class, se2name_class, se2combo = read_decagon()
smiles = read_smiles()
drugs, proteins, se_mono, ddi_se = read_ordered_lists()
mono_se_adj, dp_adj, mol_embed, ddi_adj = get_drug_features()
train_pairs, train_y, val_pairs, val_y, test_pairs, test_y = get_tvt()

train_Y = [np.array([int(v) for v in se]) for se in train_y]
test_Y = [np.array([int(v) for v in se]) for se in test_y]
val_Y = [np.array([int(v) for v in se]) for se in val_y]

# get x and y input for model

# dimensionality reduction
pca_val = 0.95

# create PCA model for mono se
mono_pca = PCA(pca_val)
mono_pca.fit(mono_se_adj)
mono_se_pca_vector = mono_pca.transform(mono_se_adj)

# create PCA model for dp 
dp_pca = PCA(pca_val)
dp_pca.fit(dp_adj)
dp_pca_vector = dp_pca.transform(dp_adj)

# create full drug features
total_features = np.concatenate((mono_se_pca_vector, dp_pca_vector, mol_embed), axis=1)

print(f'Drug Features for PCA({pca_val}): {total_features.shape}')
del mono_se_pca_vector, dp_pca_vector

# ------------------------------------  Model Training ----------------------------------------

summary_for_model = []

run_type = 'drivenn_training'
methods = 'all_dps'

for i in range(len(ddi_se)):
    train_x, val_x, test_x = single_data_split(total_features, i, \
                                           train_pairs, val_pairs, test_pairs, \
                                           drugs, ddi_adj)
    start = time.time()
    try:
        scores = train_single_model(f'{run_type}/{methods}', i, ddi_se, \
                                train_x, train_Y[i], val_x, val_Y[i], test_x, test_Y[i], \
                                se2name)
    except Exception as e:
        print("issue with model " + str(i) + ": ", e)
        scores = [i, 0, 0, 0, 0, 0, 0, 0]
    stop = time.time()
    duration = stop-start
    scores.append(duration)
    summary_for_model.append(scores)

#save results
results = pd.DataFrame(summary_for_model, columns=["se", "roc", "aupr", "precision", "recall", "f_score", "acc", "mcc", "duration"])
results.to_csv(f'training/trained_models/model_scores/{run_type}_{methods}.csv', index=False, float_format='%.5f')

print("Results DF saved to " + f'training/trained_models/model_scores/{run_type}_{methods}.csv')