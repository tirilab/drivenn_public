# logging utils
import wandb

# model utils
from keras.layers.core import Dropout, Activation
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.models import load_model

# data utils
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
import random
from operator import add
import scipy.sparse as sp
from sklearn import metrics
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utility import *

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
	se2combo = np.load('../data/model_data/se2combo.npy',allow_pickle='TRUE').item()
	return combo2stitch, combo2se, se2name, net, node2idx, stitch2se, se2name_mono, stitch2proteins, se2class, se2name_class, se2combo

def read_smiles():
	# read in smiles data
	smiles = pd.read_csv("../data/SMILES/id_SMILE.txt", sep="\t", header=None)

	# format smiles data to have same form as CID0*
	smiles.iloc[:, 0] = [f'CID{"0"*(9-len(str(drug_id)))}{drug_id}' for drug_id in smiles.iloc[:,0]]
	smiles = smiles.rename(columns={0:'drug_id', 1:'smiles'})

	return smiles

def read_ordered_lists():
	# get list of all drugs
	drugs = list(pd.read_csv('../data/model_data/drugs_ordered.csv')['drugs'])

	# get list of all proteins
	proteins = list(pd.read_csv('../data/model_data/proteins_ordered.csv')['proteins'])

	# get list of all mono se
	se_mono = list(pd.read_csv('../data/model_data/se_mono_ordered.csv')['se_mono'])

	# get list of all ddi se
	ddi_se = list(pd.read_csv('../data/model_data/ddi_se_ordered.csv')['ddi types'])

	return drugs, proteins, se_mono, ddi_se

def read_cvd_lists():
    # get list of one cvd drugs
    one_cvd = list(pd.read_csv('../data/model_data/one_cvd_drugs_ordered.csv')['cvd_drugs'])

    # get list of two cvd drugs
    two_cvd = list(pd.read_csv('../data/model_data/two_cvd_drugs_ordered.csv')['two_cvd_drugs'])

    # get list of cvd ddis
    one_cvd_ddi_se = list(pd.read_csv('../data/model_data/one_cvd_ddi_se_ordered.csv')['cvd_ddi_se'])

    # get list of cvd ddis
    two_cvd_ddi_se = list(pd.read_csv('../data/model_data/two_cvd_ddi_se_ordered.csv')['cvd_ddi_se'])

    return one_cvd, two_cvd, one_cvd_ddi_se, two_cvd_ddi_se

def get_drug_features():
	mono_se_adj = np.load('../data/model_data/embeddings/drug_label.npy')
	dp_adj = np.load('../data/model_data/embeddings/dp_adj.npy')
	mol_embed = np.load('../data/model_data/embeddings/mol_adj.npy')
	drug_feat = np.concatenate((mono_se_adj, dp_adj, mol_embed), axis=1)

	container = np.load('../data/model_data/embeddings/ddi_adj.npy.npz')
	ddi_adj = [container[key] for key in container]

	return mono_se_adj, dp_adj, mol_embed, drug_feat, ddi_adj

def get_drug_pairs():
	edges = np.load('../data/model_data/pos_and_neg_edges_dict.npy', allow_pickle="TRUE").item()
	labels = np.load('../data/model_data/pos_and_neg_edges_labels_dict.npy', allow_pickle="TRUE").item()

	return edges, labels

def get_train_test_valid():
	container = np.load('../data/model_data/TTS/train_x.npy.npz')
	all_X_train = [container[key] for key in container]
	container = np.load('../data/model_data/TTS/train_y.npy.npz')
	all_y_train = [container[key] for key in container]

	container = np.load('../data/model_data/TTS/test_x.npy.npz')
	all_X_test = [container[key] for key in container]
	container = np.load('../data/model_data/TTS/test_y.npy.npz')
	all_y_test = [container[key] for key in container]


	container = np.load('../data/model_data/TTS/valid_x.npy.npz')
	all_X_valid = [container[key] for key in container]
	container = np.load('../data/model_data/TTS/valid_y.npy.npz')
	all_y_valid = [container[key] for key in container]

	return all_X_train, all_y_train, all_X_test, all_y_test, all_X_valid, all_y_valid

# ------------------------------- model training functions ------------------------------

# helper fns 

def get_dp_for_single_se(k, edges_all):
    # train valid test split for drug pairs per side effect
    val=[]
    test=[]
    train=[]

    a = len(edges_all[k])//10
    print(f'se num: {k} | test and val size: {a} | train size {len(edges_all[k]) - 2*a}')
    np.random.shuffle(edges_all[k])
    val.append(edges_all[k][:a])
    test.append(edges_all[k][a:a+a])
    train.append(edges_all[k][a+a:])

    return train[0], test[0], val[0]


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

def construct_model(i_dim, model_type="NNPS"):
    if model_type == "NNPS":
        #construct model
        model = Sequential()
        model.add(Dense(input_dim=i_dim, kernel_initializer='glorot_normal',units=300))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dropout(0.1))
        model.add(Dense(input_dim=300, kernel_initializer='glorot_normal', units=200))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dropout(0.1))
        model.add(Dense(input_dim=200, kernel_initializer='glorot_normal', units=100))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dropout(0.1))
        model.add(Dense(input_dim=100, kernel_initializer='glorot_normal', units=1))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))

    elif model_type == "DRIVEN":
        print("creating DRIVEN model")
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

    sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def single_data_split(drug_feat, ddi_index, edges_all, drugs, dd_adj_list):
    train, test, val = get_dp_for_single_se(ddi_index, edges_all)

    # train valid test split for drug features per side effect
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    vx, vy = [], []
    for i in val:
        d1, d2 = drugs.index(i[0]), drugs.index(i[1])
        vx.append(drug_feat[d1] + drug_feat[d2])
        vy.append(dd_adj_list[ddi_index][d1, d2])
    val_x = np.array(vx)
    val_y = np.array(vy)

    tx, ty = [], []
    for i in test:
        d1, d2 = drugs.index(i[0]), drugs.index(i[1])
        tx.append(drug_feat[d1] + drug_feat[d2])
        ty.append(dd_adj_list[ddi_index][d1, d2])
    test_x = np.array(tx)
    test_y = np.array(ty)

    trx, tr_y = [], []
    for i in train:
        d1, d2 = drugs.index(i[0]), drugs.index(i[1])
        trx.append(drug_feat[d1] + drug_feat[d2])
        tr_y.append(dd_adj_list[ddi_index][d1, d2])
        
    train_x = np.array(trx)
    train_y = np.array(tr_y)

    return train_x, train_y, val_x, val_y, test_x, test_y

def train_single_model(name, ddi_index, ddi_se, train_x, train_y, val_x, val_y, test_x, test_y, se2name, epoch=50, model_type='NNPS'):
    #get criteria
    roc_score, aupr_score, f_score, thr =[], [], [], []
    precision, recall, tpos, fpos, tneg, fneg=[], [], [], [], [], []
    acc, mcc = [], []
    
    k = ddi_index
    se = ddi_se[k]
    print(f'On model for side effect {k}: {se2name[se]}')

    model = construct_model(train_x.shape[1], model_type) 

    model.fit(train_x, train_y,batch_size=1024, epochs=epoch)
    print("done fitting model")

    val_loss, val_acc = model.evaluate(val_x, val_y)
    print("done validating model")
    print('Val accuracy:', val_acc)


    test_loss, test_acc = model.evaluate(test_x, test_y)
    print("done testing model")
    print('Test accuracy:', test_acc)
    
    test_pred = model.predict(test_x)

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
    
    wandb.log({"side_effect": k, "val_acc": val_acc, "test_acc": test_acc,"roc": roc, "aupr": aupr, "f_score": ma, "precision": p, "recall": r, "acc": ac, "matthews_corr_coef": mc})
    
    model.save(f'trained_models/{name}/model_{k}.h5')

    return [se, roc_score, aupr_score, precision, recall, f_score, acc, mcc]

def start_wandb(project_name, feature_selection):
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": "DrIVeN",
        "dataset": "DECAGON + Molecule",
        "epochs": 50,
        "feature_selection": feature_selection
        }
    )






