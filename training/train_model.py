import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import time
from tensorflow.keras.models import load_model
import argparse

from training import *
from utility import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process data with a specified random seed.')
parser.add_argument('--seed', type=int, default=13, help='Random seed for reproducibility')
parser.add_argument('--run_type', type=str, default='drivenn_training', help='Run type for clarity - model name')
parser.add_argument('--methods', type=str, default='all_dps', help='Methods for clarity - input')
parser.add_argument('--pca', type=int, default=0.95, help='PCA variance, default is 0.95')

args = parser.parse_args()
seed = args.seed
run_type = args.run_type
methods = args.methods
pca_val = args.pca

# read in saved data
combo2stitch, combo2se, se2name, net, node2idx, stitch2se, se2name_mono, stitch2proteins, se2class, se2name_class, se2combo = read_decagon()
smiles = read_smiles()
drugs, proteins, se_mono, ddi_se = read_ordered_lists()
mono_se_adj, dp_adj, mol_embed, ddi_adj = get_drug_features()
train_pairs, train_y, val_pairs, val_y, test_pairs, test_y = get_tvt(seed)

train_Y = [np.array([int(v) for v in se]) for se in train_y]
test_Y = [np.array([int(v) for v in se]) for se in test_y]
val_Y = [np.array([int(v) for v in se]) for se in val_y]

# dimensionality reduction

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

for i in range(len(ddi_se)):
    if i%100 == 0:
        print(f'On model for side effect {i}: {se2name[ddi_se[i]]}')  
    train_x, val_x, test_x = single_data_split(total_features, i, train_pairs, val_pairs, test_pairs, drugs, ddi_adj)
    model_path = f'training/trained_models/{run_type}/{methods}/{seed}/model_{i}.keras'
    start = time.time()
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)  
            scores = [ddi_se[i]] + evaluate_model(model, test_x, test_Y[i])  
        else:
            print(f"Training model {i}")
            scores = train_single_model(f'{run_type}/{methods}/{seed}', i, ddi_se, \
                                train_x, train_Y[i], val_x, val_Y[i], test_x, test_Y[i], \
                                se2name)
    except Exception as e:
        print(f"issue with model {i}")
        scores = [i, 0, 0, 0, 0, 0, 0, 0]

    stop = time.time()
    duration = stop-start
    scores.append(duration)
    summary_for_model.append(scores)

#save results
results = pd.DataFrame(summary_for_model, columns=["se", "roc", "aupr", "precision", "recall", "f_score", "acc", "mcc", "duration"])
results.to_csv(f'training/trained_models/model_scores/{run_type}_{methods}_{seed}.csv', index=False, float_format='%.5f')

print("Results DF saved to " + f'training/trained_models/model_scores/{run_type}_{methods}_{seed}.csv')