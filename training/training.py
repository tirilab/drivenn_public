# model utils
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation

# data utils
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from training.utility import * 
import os
import csv

# this file should contain all the functions needed to train

# ------------------------------- read in preprocessed data ------------------------------

def read_decagon():
	# read in decagon data
	combo2stitch, combo2se, se2name = load_combo_se()
	net, node2idx = load_ppi()
	stitch2se, se2name_mono = load_mono_se()
	stitch2proteins = load_targets()
	se2class, se2name_class = load_categories()
	se2name.update(se2name_mono)
	se2name.update(se2name_class)
	se2combo = np.load('data/model_data/embeddings/se2combo.npy',allow_pickle='TRUE').item()
	return combo2stitch, combo2se, se2name, net, node2idx, stitch2se, se2name_mono, stitch2proteins, se2class, se2name_class, se2combo

def read_smiles():
	# read in smiles data
	smiles = pd.read_csv("data/model_data/SMILES/id_SMILE.txt", sep="\t", header=None, dtype="object")

	# format smiles data to have same form as CID0*
	smiles.iloc[:, 0] = [f'CID{"0"*(9-len(str(drug_id)))}{drug_id}' for drug_id in smiles.iloc[:,0]]
	smiles = smiles.rename(columns={0:'drug_id', 1:'smiles'})

	return smiles

def read_ordered_lists():
	# get list of all drugs
	drugs = list(pd.read_csv('data/model_data/drugs_ordered.csv')['drugs'])

	# get list of all proteins
	proteins = list(pd.read_csv('data/model_data/proteins_ordered.csv')['proteins'])

	# get list of all mono se
	se_mono = list(pd.read_csv('data/model_data/se_mono_ordered.csv')['se_mono'])

	# get list of all ddi se
	ddi_se = list(pd.read_csv('data/model_data/ddi_se_ordered.csv')['ddi types'])

	return drugs, proteins, se_mono, ddi_se

def read_cvd_lists():
    # get list of one cvd drugs
    one_cvd = list(pd.read_csv('data/model_data/one_cvd_drugs_ordered.csv')['drugs'])

    # get list of cvd ddis
    one_cvd_ddi_se = list(pd.read_csv('data/model_data/one_cvd_ddi_se_ordered.csv')['cvd_ddi_se'])

    return one_cvd, one_cvd_ddi_se

def get_drug_features():
	mono_se_adj = np.load('data/model_data/embeddings/drug_label.npy')
	dp_adj = np.load('data/model_data/embeddings/dp_adj.npy')
	mol_embed = np.load('data/model_data/embeddings/mol_adj.npy')

	container = np.load('data/model_data/embeddings/ddi_adj.npz')
	ddi_adj = [container[key] for key in container]

	return mono_se_adj, dp_adj, mol_embed, ddi_adj

def get_cvd_drug_features():
	one_cvd_mono_se_adj = np.load('data/model_data/embeddings/cvd_drug_label.npy')
	one_cvd_dp_adj = np.load('data/model_data/embeddings/cvd_dp_adj.npy')
	one_cvd_mol_embed = np.load('data/model_data/embeddings/cvd_mol_adj.npy')

	container = np.load('data/model_data/embeddings/one_cvd_ddi_adj.npz')
	one_cvd_ddi_adj = [container[key] for key in container]

	return one_cvd_mono_se_adj, one_cvd_dp_adj, one_cvd_mol_embed, one_cvd_ddi_adj

def get_tvt():
    with open("data/model_data/TTS/test_pairs.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        test_pairs = list(csv_reader)
    with open("data/model_data/TTS/test_y.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        test_y = list(csv_reader)
        test_Y = [[int(v) for v in se] for se in test_y] 
    with open("data/model_data/TTS/val_pairs.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        val_pairs = list(csv_reader)
    with open("data/model_data/TTS/val_y.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        val_y = list(csv_reader)
        val_Y = [[int(v) for v in se] for se in val_y]
    with open("data/model_data/TTS/train_pairs.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        train_pairs = list(csv_reader)
    with open("data/model_data/TTS/train_y.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        train_y = list(csv_reader)
        # train_Y = [[int(v) for v in se] for se in train_y]
    return train_pairs, train_y, val_pairs, val_Y, test_pairs, test_Y

def get_cvd_tvt():
    with open("data/model_data/TTS/cvd/test_pairs.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        test_pairs = list(csv_reader)
    with open("data/model_data/TTS/cvd/test_y.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        test_y = list(csv_reader)
        test_Y = [[int(v) for v in se] for se in test_y] 
    with open("data/model_data/TTS/cvd/val_pairs.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        val_pairs = list(csv_reader)
    with open("data/model_data/TTS/cvd/val_y.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        val_y = list(csv_reader)
        val_Y = [[int(v) for v in se] for se in val_y]
    with open("data/model_data/TTS/cvd/train_pairs.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        train_pairs = list(csv_reader)
    with open("data/model_data/TTS/cvd/train_y.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        train_y = list(csv_reader)
        train_Y = [[int(v) for v in se] for se in train_y]
    return train_pairs, train_Y, val_pairs, val_Y, test_pairs, test_Y

# ------------------------------- model training functions ------------------------------

# helper fns 
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def construct_model(i_dim):
    # print("creating DRIVENN model")
    #construct model
    model = Sequential()
    model.add(Dense(input_dim=i_dim, kernel_initializer='glorot_normal',units=300))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(input_dim=300, kernel_initializer='glorot_normal', units=100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(input_dim=100, kernel_initializer='glorot_normal', units=1))
    model.add(Activation('sigmoid'))

    sgd = optimizers.legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def single_data_split(drug_feat, ddi_index, train_dp, val_dp, test_dp, drugs, dd_adj_list):
    # train valid test split for drug features per side effect
    train_x, val_x, test_x = [], [], []

    vx = []
    for i in val_dp[ddi_index]:
        if "_" in i:
            i = i.split("_")
        d1, d2 = drugs.index(i[0]), drugs.index(i[1])
        vx.append(np.concatenate((drug_feat[d1], drug_feat[d2])))
    val_x = np.array(vx)
    
    tx = []
    for i in test_dp[ddi_index]:
        if "_" in i:
            i = i.split("_")
        d1, d2 = drugs.index(i[0]), drugs.index(i[1])
        tx.append(np.concatenate((drug_feat[d1], drug_feat[d2])))
    test_x = np.array(tx)
    
    trx = []
    for i in train_dp[ddi_index]:
        if "_" in i:
            i = i.split("_")
        d1, d2 = drugs.index(i[0]), drugs.index(i[1])
        trx.append(np.concatenate((drug_feat[d1], drug_feat[d2])))
    train_x = np.array(trx)
    

    return train_x, val_x, test_x

def train_single_model(name, ddi_index, ddi_se, train_x, train_y, val_x, val_y, test_x, test_y, se2name, epoch=50):
    #get criteria
    roc_score, aupr_score, f_score, thr =[], [], [], []
    precision, recall, tpos, fpos, tneg, fneg=[], [], [], [], [], []
    acc, mcc = [], []
    
    k = ddi_index
    se = ddi_se[k]
    if k%100 == 0:
        print(f'On model for side effect {k}: {se2name[se]}')

    model = construct_model(train_x.shape[1]) 

    model.fit(train_x, train_y,batch_size=1024, epochs=epoch, verbose=0)

    # val_loss, val_acc = model.evaluate(val_x, val_y)
    # print("done validating model")
    # print('Val accuracy:', val_acc)


    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    test_pred = model.predict(test_x, verbose=0)

    roc=metrics.roc_auc_score(test_y, test_pred)
    roc_score.append(roc)
    aupr=metrics.average_precision_score(test_y,test_pred)
    aupr_score.append(aupr)

    fpr, tpr, thresholds=metrics.roc_curve(test_y,test_pred)
    scores=[metrics.f1_score(test_y, to_labels(test_pred, t)) for t in thresholds]
    ma= max(scores)
    f_score.append(ma)
    idx=np.argmax(scores)
    bt=thresholds[idx]
    thr.append(bt)

    p=metrics.precision_score(test_y, to_labels(test_pred, bt))
    precision.append(p)

    r=metrics.recall_score(test_y, to_labels(test_pred, bt))
    recall.append(r)

    TP, FP, TN, FN=perf_measure(test_y,to_labels(test_pred, bt))
    tpos.append(TP)
    fpos.append(FP)
    tneg.append(TN)
    fneg.append(FN)

    ac = float(TP + TN)/(TP+FP+FN+TN)
    acc.append(ac)

    mc=metrics.matthews_corrcoef(test_y,to_labels(test_pred, bt))
    mcc.append(mc)

    if not os.path.exists(f'training/trained_models/{name}/'):
        os.makedirs(f'training/trained_models/{name}/')
    model.save(f'training/trained_models/{name}/model_{k}.keras')

    return [se, roc, aupr, p, r, ma, ac, mc]






