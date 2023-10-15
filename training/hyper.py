from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
import csv

# read in utils
from utility import *
from training import *
import time

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

# ---------------------------------  get drug features --------------------------------- 

# mono se features
drug_label = np.zeros((n_drugs,len(se_mono)))
for key,value in stitch2se.items():
	j = drugs.index(key)
	for v in value:
		i=se_mono.index(v)
		drug_label[j,i]=1
		
# drug protein features
dp_adj = np.zeros((n_drugs, n_proteins))
for i in range(n_drugs):
	drug_name = drugs[i]
	for protein_name in stitch2proteins[drug_name]:
		k = proteins.index(int(protein_name))
		dp_adj[i, k] = 1

# read in drug molecule features (embeddings from dgllife)
mol_adj  = np.load('../data/model_data/embeddings/mol_adj.npy')


drug_feat = np.concatenate((drug_label, dp_adj, mol_adj), axis=1)

print("full drug feature shape: ", drug_feat.shape)

# --------------------------------- dd adj matrix --------------------------------- 

# format drug drug adjacency matrix for each side effect
dd_adj_list = []

for i in range(len(ddi_se)):
	if i%100 == 0:
		print(f'SE number: {i}')
	mat = np.zeros((n_drugs, n_drugs))
	curr_se = ddi_se[i]
	drug_pairs_for_se = se2combo[curr_se]
	for dp in drug_pairs_for_se:
		d1, d2 = dp.split("_")
		drug1 = drugs.index(d1)
		drug2 = drugs.index(d2)
		mat[drug1, drug2] = 1
		mat[drug2, drug1] = 1
	dd_adj_list.append(mat)

print('finished creating drug drug adj matrix for each side effect')

# --------------------------------- get pos and neg edges --------------------------------- 

# get positive and negative edges
edges_all = []

for i in range(len(ddi_se)):
	if i%100 == 0:
		print(f'SE number: {i}')
	curr_se = ddi_se[i]
	curr_edges = se2combo[curr_se]
	curr_edges_set = set(curr_edges)
	c = [pair.split("_") for pair in curr_edges]
	
	# create an equal number of neg edges
	n = []
	while len(n) < len(curr_edges):
		d1, d2 = random.choice(drugs), random.choice(drugs)
		rand_pair, rand_pair_inv = f'{d1}_{d2}', f'{d2}_{d1}'
		if rand_pair not in curr_edges_set and rand_pair_inv not in curr_edges_set:
			n.append([d1, d2])
	
	# create edges_all
	all_edges = c + n
	np.random.shuffle(all_edges)
	edges_all.append(all_edges[:len(c)])


del c
del n

print('finished getting all edges')

# --------------------------------- get input features --------------------------------- 
# pca_val = 0.95

# # create PCA model for mono se
# mono_pca = PCA(pca_val)
# mono_pca.fit(drug_label)
# mono_se_pca_vector = mono_pca.transform(drug_label)

# # create PCA model for dp 
# dp_pca = PCA(pca_val)
# dp_pca.fit(dp_adj)
# dp_pca_vector = dp_pca.transform(dp_adj)

# # create full drug features
# total_features = np.concatenate((mono_se_pca_vector, dp_pca_vector, mol_adj), axis=1)
# print(f'Drug Features for PCA({pca_val}): {total_features.shape}')
# del mono_se_pca_vector, dp_pca_vector

# np.save("data/hyper_total_features.npy", total_features)

total_features = np.load("../data/hyper_total_features.npy")
print(f'Drug Features for PCA(0.95) for Hyper Param Training: {total_features.shape}')

# --------------------------------- get data splits for each se and start hyperparam tuning --------------------------------- 

for i in range(len(ddi_se)//2):
	print(f'On SE {i}: {se2name[ddi_se[i]]}')
	train_x, train_y, val_x, val_y, test_x, test_y = single_data_split(total_features, i, edges_all, drugs, dd_adj_list)
	print(train_x.shape, val_x.shape, test_x.shape)

	def build_model(hp):
	    model = keras.Sequential()
	    model.add(layers.Dense(units=hp.Int('units_' + str(1), min_value=100, max_value = 300, step=100),
	                                  activation='relu', input_dim = 825))
	    if hp.Boolean("dropout"):
	                model.add(layers.Dropout(hp.Choice(name='dropout_val_', values=[0.1, 0.3, 0.5, 0.8, 0.9])))
	    if hp.Boolean("batchnorm"):
	        model.add(layers.BatchNormalization())
					
	    for i in range(hp.Int('num_layers', 1, 3)):
	        if i > 1:
	            model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=100, max_value = 300, step=100),
	                                      activation='relu'))
	            if hp.Boolean("dropout"):
	                model.add(layers.Dropout(hp.Choice(name='dropout_val_' , values=[0.1, 0.3, 0.5, 0.8, 0.9])))
	            if hp.Boolean("batchnorm"):
	                model.add(layers.BatchNormalization())

	    model.add(layers.Dense(1, activation='sigmoid'))

	    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
	    return model

	build_model(keras_tuner.HyperParameters())

	tuner = keras_tuner.Hyperband(
		hypermodel=build_model,
		objective="val_accuracy",
		max_epochs=20,
		directory="hyperparam_all",
		project_name=f'hyperband_{i}',
	)

	tuner.search_space_summary()

	tuner.search(train_x, train_y, epochs=50, validation_data=(val_x, val_y))

	








