from utility import *

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
import collections
import json as json

# load dataset
combo2stitch, combo2se, se2name = load_combo_se(fname='../data/decagon_data/bio-decagon-combo.csv')
net, node2idx = load_ppi(fname='../data/decagon_data/bio-decagon-ppi.csv')
stitch2se, se2name_mono = load_mono_se(fname='../data/decagon_data/bio-decagon-mono.csv')
stitch2proteins = load_targets(fname='../data/decagon_data/bio-decagon-targets.csv')
se2class, se2name_class = load_categories(fname='../data/decagon_data/bio-decagon-effectcategories.csv')
se2name.update(se2name_mono)
se2name.update(se2name_class)

# summary of original dataset

# drugs with protein info
drugs_w_protein = list(stitch2proteins.keys())
print("drug proteins: " + str(len(drugs_w_protein)))

# drugs with individual SE info 
drugs_w_indiv_se = list(stitch2se.keys())
print("indiv drugs: " + str(len(drugs_w_indiv_se)))

# drugs with combo SE info
combo_drugs = np.unique(np.array([i for pair in list(combo2stitch.values()) for i in pair]))
print("combo drugs: " + str(len(combo_drugs)))

# total drugs
total_drugs = []
[total_drugs.extend(li) for li in (drugs_w_protein, drugs_w_indiv_se, combo_drugs)]
total_drugs = np.array(total_drugs)
total_drugs = np.unique(np.array(total_drugs))
print("total unique drugs: " + str(len(total_drugs)))


# read in cvd drugs from NCATS
mi_df = pd.read_csv('../data/NCATS_exports/export_all_uid_MI.tsv', sep='\t')
cad_df = pd.read_csv('../data/NCATS_exports/export_all_uid_CAD.tsv', sep='\t')
chf_df = pd.read_csv('../data/NCATS_exports/export_all_uid_CHF.tsv', sep='\t')

# read in UNII records from GSRS
unii_records = pd.read_csv("../data/UNII_Data/UNII_Records_13Apr2023.txt", sep='\t', low_memory=False)
legacy_unii_records = pd.read_csv("../data/UNII_Data/Legacy UNIIs.txt", sep='\t', low_memory=False)

# merge UNII records with total drugs and cvd drug lists
total_df = pd.DataFrame({"total_drugs": total_drugs,
             "drug_num": [int(d[3::].lstrip("0")) for d in total_drugs]})

total_merged = pd.merge(total_df, unii_records, how='left', right_on = 'PUBCHEM', left_on='drug_num')
total_merged = total_merged[["total_drugs", "drug_num", "UNII"]]

# get drugs with no matched UNIIs to look up manually
null_drugs = total_merged[total_merged['UNII'].isnull()]
null_drugs.to_csv("../data/UNII_Data/null_drugs.csv")

# DO THIS AFTER YOU LOOK UP THE UNMATCHED DRUGS - OR JUST USE THE PROVIDED manual_null_drugs.csv
manual = pd.read_csv("../data/UNII_Data/manual_null_drugs.csv")

# get the number of cvd drugs found in total drugs
total_merged = pd.merge(total_merged, manual[["total_drugs", "UNII"]], 
         on = "total_drugs", how = "left")

total_merged["UNII"] = [i if i is not np.nan else j for i, j in zip(total_merged.UNII_x, total_merged.UNII_y)]
total_merged = total_merged[["total_drugs", "drug_num", "UNII"]]

mi_merged = pd.merge(total_merged, mi_df, on = 'UNII', how='inner')
cad_merged = pd.merge(total_merged, cad_df, on = 'UNII', how='inner')
chf_merged = pd.merge(total_merged, chf_df, on = 'UNII', how='inner')

# create and save cvd drug df
cvd = pd.concat([mi_df, cad_df, chf_df])
cvd_df = pd.merge(total_merged, cvd, on = 'UNII', how='inner').drop_duplicates("UNII").reset_index(drop=True)
cvd_df.to_csv("../data/cvd_df.csv")

# format smiles for next step
smile = pd.read_csv("../data/SMILES/id_SMILE.txt", sep='\t', header=None)
smile.columns = ["ID", "SMILE"]
smile['ID'] = smile['ID'].apply(lambda x: f'Drug::{x}')

smile.to_csv("../data/SMILES/drugname_smiles.txt", index=None, sep="\t")



