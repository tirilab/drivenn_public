from collections import defaultdict
import networkx as nx
import pandas as pd

# Returns dictionary from combination ID to pair of stitch IDs, 
# dictionary from combination ID to list of polypharmacy side effects, 
# and dictionary from side effects to their names.
def load_combo_se(fname="data/decagon_data/bio-decagon-combo.csv"):
    combo2stitch = {}
    combo2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    # print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id1, stitch_id2, se, se_name = line.strip().split(',')
        combo = stitch_id1 + '_' + stitch_id2
        combo2stitch[combo] = [stitch_id1, stitch_id2]
        combo2se[combo].add(se)
        se2name[se] = se_name
    fin.close()
    n_interactions = sum([len(v) for v in combo2se.values()])
    # print('Drug combinations: %d Side effects: %d' % (len(combo2stitch), len(se2name)))
    # print('Drug-drug interactions: %d' % (n_interactions))
    return combo2stitch, combo2se, se2name

# Returns networkx graph of the PPI network 
# and a dictionary that maps each gene ID to a number
def load_ppi(fname='data/decagon_data/bio-decagon-ppi.csv'):
    fin = open(fname)
    # print('Reading: %s' % fname)
    fin.readline()
    edges = []
    for line in fin:
        gene_id1, gene_id2= line.strip().split(',')
        edges += [[gene_id1,gene_id2]]
    nodes = set([u for e in edges for u in e])
    # print('Edges: %d' % len(edges))
    # print('Nodes: %d' % len(nodes))
    net = nx.Graph()
    net.add_edges_from(edges)
    net.remove_nodes_from(nx.isolates(net))
    net.remove_edges_from(nx.selfloop_edges(net))
    node2idx = {node: i for i, node in enumerate(net.nodes())}
    return net, node2idx

# Returns dictionary from Stitch ID to list of individual side effects, 
# and dictionary from side effects to their names.
def load_mono_se(fname='data/decagon_data/bio-decagon-mono.csv'):
    stitch2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    # print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        contents = line.strip().split(',')
        stitch_id, se, = contents[:2]
        se_name = ','.join(contents[2:])
        stitch2se[stitch_id].add(se)
        se2name[se] = se_name
    return stitch2se, se2name

# Returns dictionary from Stitch ID to list of drug targets
def load_targets(fname='data/decagon_data/bio-decagon-targets-all.csv'):
    stitch2proteins = defaultdict(set)
    fin = open(fname)
    # print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id, gene = line.strip().split(',')
        stitch2proteins[stitch_id].add(gene)
    return stitch2proteins

# Returns dictionary from side effect to disease class of that side effect,
# and dictionary from side effects to their names.
def load_categories(fname='data/decagon_data/bio-decagon-effectcategories.csv'):
    se2name = {}
    se2class = {}
    fin = open(fname)
    # print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        se, se_name, se_class = line.strip().split(',')
        se2name[se] = se_name
        se2class[se] = se_class
    return se2class, se2name

def read_decagon():
	# read in decagon data
	combo2stitch, combo2se, se2name = load_combo_se()
	net, node2idx = load_ppi()
	stitch2se, se2name_mono = load_mono_se()
	stitch2proteins = load_targets()
	se2class, se2name_class = load_categories()
	se2name.update(se2name_mono)
	se2name.update(se2name_class)
	return combo2stitch, combo2se, se2name, net, node2idx, stitch2se, se2name_mono, stitch2proteins, se2class, se2name_class

def read_smiles():
	# read in smiles data
	smiles = pd.read_csv("data/model_data/SMILES/id_SMILE.txt", sep="\t", header=None, dtype="object")

	# format smiles data to have same form as CID0*
	smiles.iloc[:, 0] = [f'CID{"0"*(9-len(str(drug_id)))}{drug_id}' for drug_id in smiles.iloc[:,0]]
	smiles = smiles.rename(columns={0:'drug_id', 1:'smiles'})

	return smiles