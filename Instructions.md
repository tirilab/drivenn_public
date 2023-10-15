# Data

1. decagon_data - download project files from [SNAP](http://snap.stanford.edu/decagon/).

2. UNII_Data - download UNII substance data from [GSRS](https://precision.fda.gov/uniisearch).

3. ADR_severity_data - download SAEDR_scores as described in this [paper by Lavertu et. al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8569532/).

# Data Processing

1. Get drugs indicated for CVD diseases using NCATS_indications.py. Update the 'facet' and 'name' variables in the script with the disease you want to search in NCATS API. Outputs to ```../data/NCATS_exports/```.

    ```python3 NCATS_indications.py```

2. Create your CVD dataset by matching UNII records and manually looking up the rest of the drugs.

    ```python3 cvd_dataset_creation.py```

3. Get your graph-based smiles drug embeddings.

    ```python pretrain_smiles_embedding.py -fi ../data/SMILES/drugname_smiles.txt -m gin_supervised_masking -fo csv -sc SMILE -o ../data/model_data/embeddings/```

4. EDA example in notebook ```EDA.ipynb```

# Training

1. Feature selection between PCA (0.85, 0.9, 0.95, 0.99) and UMAP.

    ```python3 feature_selection.py```

2. Hyperparameter tuning with Hyperband looking at layers, dropout, batchnorm.

    ```python3 hyper.py```

3. CVD model training. You can update feature selection method in script if you want to. Default is PCA(0.95). Outputs to ```../data/cvd_output/```.

    ```python3 cvd_training.py```

# Severity Analysis

1. Analysis example in ```Severity_Analysis.ipynb```.
