# Set Up Environment

1. Clone GitHub repo and enter it.
   
   ```git clone https://github.com/tirilab/drivenn_public.git```
   
   ```cd drivenn_public```

2. Create and activate a conda virtual environment with python 3.10.14.

   ```conda create -n <venv_name> python=3.10.14```
   
   ```conda activate <venv_name>```

3.  Install pip.

    ```conda install pip```

4.  Install requirements.txt.

    ```pip install -r requirements.txt```

# Data Files

1. decagon_data: Download all project files from [SNAP](http://snap.stanford.edu/decagon/) and unzip by right clicking on each one. Place in ```data/decagon_data```.
    
    Saved Files:
    - ```data/decagon_data/bio-decagon-combo.csv```
    - ```data/decagon_data/bio-decagon-effectcategories.csv```
    - ```data/decagon_data/bio-decagon-mono.csv```
    - ```data/decagon_data/bio-decagon-ppi.csv```
    - ```data/decagon_data/bio-decagon-targets-all.csv```
    - ```data/decagon_data/bio-decagon-targets.csv```

2. UNII_Data: Download UNII substance data (Legacy UNIIs and UNII Data) from [GSRS](https://precision.fda.gov/uniisearch/archive). Unzip by right clicking, rename, and place in ```data/UNII_Data/```.

    Saved Files:
    -  ```data/UNII_Data/UNII_Records.txt```
    - ```data/UNII_Data/Legacy_UNIIs.txt``` 

3. ADR_severity_data: Download SAEDR_scores as described in this [paper by Lavertu et. al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8569532/) in Multimedia Appendix 2. Unzip by right clicking on it and place in ```data/ADR_severity_data/```.
    Saved Files:
    - ```data/ADR_severity_data/```

# Data Processing

1. Get drugs indicated for CVD diseases (CHF, MI, CAD) using NCATS_indications.py. If you wish to look up other diseases you can update 'facet' and 'name' variables in the script with the disease you want to search in NCATS API. 

    ```python3 data_processing/NCATS_indications.py```

    Output Files: 
    - ```data/NCATS_exports/export_all_uid_CHF.tsv``` (tsv of Congestive Heart Failure indicated drugs)
    - ```data/NCATS_exports/export_all_uid_CAD.tsv``` (tsv of Coronary Artery Disease indicated drugs)
    - ```data/NCATS_exports/export_all_uid_MI.tsv``` (tsv of Myocardial Infarction indicated drugs)

2. Create your CVD dataset by matching UNII records and manually looking up the rest of the drugs. Note: we have included the list of drugs we manually looked up in ```data/UNII_Data/manual_null_drugs.csv```, you can edit or add additional drugs to it if you wish.

    ```python3 data_processing/cvd_dataset_creation.py```

    Output Files:
    - ```data/updated_cvd_df.csv``` (csv of identified CVD-related drugs containing their UNIIs, Drug Names, and SMILES) NOTE: To ensure repeatability, we have also included ```data/cvd_df.csv``` in the repo which includes the CVD drugs indicated when we ran this script in 2023. Exact drugs returned by the API may change slightly as NCATS is updated. The rest of the scripts use our file, but if you wish to run it on the updated drugs just rename your file to ```data/cvd_df.csv```.
    - ```data/model_data/SMILES/drugname_smiles.csv``` (csv of all drug ids to SMILES)

3. Get your graph-based smiles drug embeddings.

    ```python3 data_processing/pretrain_smiles_embedding.py -fi data/SMILES/drugname_smiles.txt -m gin_supervised_masking -fo csv -sc SMILE -o data/model_data/embeddings/```

    Output Files:
    -  ```data/model_data/embeddings/drugname_smiles.npy``` (file with generated smiles drug embeddings)

4. Prepare and format data for faster training and easier reproducibility.

    ```python3 data_prep.py```

    Output Files:
    - Basic Lists:
        - ```data/model_data/drugs_ordered.csv``` (ordered csv of list of drugs)
        - ```data/model_data/one_cvd_drugs_ordered.csv``` (ordered csv of list of drugs in drug pairs with at least one cvd drug)
        - ```data/model_data/proteins_ordered.csv``` (ordered csv of list of proteins)
        - ```data/model_data/se_mono_ordered.csv``` (ordered csv of list of mono se's)
        - ```data/one_cvd.csv``` (drug pairs with at least one cvd drug)
        - ```data/two_cvd.csv``` (drug pairs with 2 cvd drugs)
    - DDI Lists:
        - ```data/model_data/ddi_se_ordered.csv``` (ordered csv of list of drug-drug interactions)
        - ```data/model_data/one_cvd_ddi_se_ordered.csv``` (ordered csv of list of drug-drug interactions for drug pairs with at least one cvd drug)
        - ```data/model_data/two_cvd_ddi_se_ordered.csv``` (ordered csv of list of drug-drug interactions for drug pairs with two cvd drugs)
    - Drug Features:
        - ```data/model_data/embeddings/drug_label.npy``` (ordered mono se matrix)
        - ```data/model_data/embeddings/one_cvd_drug_label.npy``` (ordered mono se matrix for drug pairs with at least one cvd drug)
        - ```data/model_data/embeddings/dp_adj.npy``` (ordered drug-protein adjacency matrix)
        - ```data/model_data/embeddings/one_cvd_dp_adj.npy``` (ordered drug-protein adjacency matrix with at least one cvd drug)
        - ```data/model_data/embeddings/mol_adj.npy``` (ordered drug-molecule matrix)
        - ```data/model_data/embeddings/one_cvd_mol_adj.npy``` (ordered drug-molecule matrix for drug pairs with at least one cvd drug)
    - DDI Matrices
        - ```data/model_data/embeddings/ddi_adj.npy``` (ordered drug-drug interaction matrix)
        - ```data/model_data/embeddings/one_cvd_ddi_adj.npy``` (ordered drug-drug interaction matrix for drug pairs with at least one cvd drug)
    - Edges
        - ```data/model_data/edges_all``` (all edges)
        - ```data/model_data/one_cvd_edges_all``` (all edges for drug pairs with at least one cvd drug)
    - Train Validation Test Splits
        - ```data/model_data/TTS/train_dps.npz``` (train drug pairs)
        - ```data/model_data/TTS/valid_dps.npz``` (valid drug pairs)
        - ```data/model_data/TTS/test_dps.npz``` (test drug pairs)
        - ```data/model_data/TTS/one_cvd_train_dps.npz``` (train drug pairs for dataset of drug pairs with at least one cvd drug)
        - ```data/model_data/TTS/one_cvd_valid_dps.npz``` (valid drug pairs for dataset of drug pairs with at least one cvd drug)
        - ```data/model_data/TTS/one_cvd_test_dps.npz``` (test drug pairs for dataset of drug pairs with at least one cvd drug)

5. EDA example in notebook ```data_processing/EDA.ipynb```

# Training

1. Training the DrIVeNN model with the dataset. You can update feature selection method in script if you want to. Default is PCA(0.95).

    ```python3 training/train_model.py```

    Output Files:
    - ```training/trained_models/model_scores/drivenn_training_all_dps.csv``` (saved metrics from training for each model)
    - ```training/trained_models/drivenn_training/all_dps/model_{i}.keras``` (saved trained model for each se)

2. CVD model training. You can update feature selection method in script if you want to. Default is PCA(0.95). Outputs to ```../data/cvd_output/```.

    ```python3 training/cvd_training.py```

    Output Files:
    - ```training/trained_models/model_scores/cvd_drivenn_training_cvd_dps.csv``` (saved metrics from training for each model)
    - ```training/trained_models/cvd_drivenn_training/cvd_dps/model_{i}.keras``` (saved trained model for each se)

# Severity Analysis

1. Analysis example in ```analysis/Severity_Analysis.ipynb```.
