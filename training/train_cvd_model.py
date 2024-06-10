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
one_cvd, one_cvd_ddi_se = read_cvd_lists()
one_cvd_mono_se_adj, one_cvd_dp_adj, one_cvd_mol_embed, one_cvd_ddi_adj = get_cvd_drug_features()
cvd_X_train, cvd_X_valid, cvd_X_test = get_cvd_tvt()

# get x and y input for model

# dimensionality reduction
pca_val = 0.95

# create PCA model for mono se
mono_pca = PCA(pca_val)
mono_pca.fit(one_cvd_mono_se_adj)
mono_se_pca_vector = mono_pca.transform(one_cvd_mono_se_adj)

# create PCA model for dp 
dp_pca = PCA(pca_val)
dp_pca.fit(one_cvd_dp_adj)
dp_pca_vector = dp_pca.transform(one_cvd_dp_adj)

# create full drug features
total_features = np.concatenate((mono_se_pca_vector, dp_pca_vector, one_cvd_mol_embed), axis=1)

del mono_se_pca_vector, dp_pca_vector

# ------------------------------------  CVD Model Training ----------------------------------------
summary_for_model = []
run_type = 'cvd_drivenn_training'
methods = 'cvd_dps'

for i in range(len(one_cvd_ddi_se)):
    train_x, train_y, val_x, val_y, test_x, test_y = single_data_split(total_features, i, cvd_X_train, cvd_X_valid, cvd_X_test, one_cvd, one_cvd_ddi_adj)
    start = time.time()
    try:
        scores = train_single_model(f'{run_type}/{methods}', i, one_cvd_ddi_se, \
                                train_x, train_y, val_x, val_y, test_x, test_y, \
                                se2name)
    except:
        print("issue with model " + str(i))
        scores = [i, 0, 0, 0, 0, 0, 0, 0]
    stop = time.time()
    duration = stop-start
    scores.append(duration)
    summary_for_model.append(scores)

#save results
results = pd.DataFrame(summary_for_model, columns=["se", "roc", "aupr", "precision", "recall", "f_score", "acc", "mcc", "duration"])
results.to_csv(f'training/trained_models/model_scores/{run_type}_{methods}.csv', index=False, float_format='%.5f')

print("Results DF saved to " + f'training/trained_models/model_scores/{run_type}_{methods}.csv')