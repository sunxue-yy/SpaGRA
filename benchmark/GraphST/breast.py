import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
from DeepST import DeepST
import numpy as np
os.environ['R_HOME'] = '/home/dell/anaconda3/envs/stagate/lib/R'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ARIlist = []
dataset="V1_Breast_Cancer_Block_A_Section_1"
n_clusters=20
print(dataset)
input_dir = os.path.join('Data', dataset)
adata = sc.read_visium(path=input_dir, count_file=dataset + '_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

model = DeepST.DeepST(adata, device=device)
adata = model.train_DeepST()

radius = 50
from DeepST.utils import clustering
clustering(adata, n_clusters, radius=radius, refinement=True) #For DLPFC dataset, we use optional refinement step.

df_meta  = pd.read_csv("Data/V1_Breast_Cancer_Block_A_Section_1/metadata.tsv", sep="	", header=0, na_filter=False,
                          index_col=0)
adata.obs['fine_annot_type'] = df_meta.loc[adata.obs_names, 'fine_annot_type']
df_meta["chenjinmao"]=adata.obs['domain']
df_meta.to_csv("Data/graphst_" + dataset + ".csv")

# filter out NA nodes
adata = adata[~pd.isnull(adata.obs['fine_annot_type'])]

# calculate metric ARI
ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['fine_annot_type'])
adata.uns['ARI'] = ARI
ARIlist.append(ARI)
print('Dataset:', dataset)
print('ARI:', ARI)

from sklearn.metrics import normalized_mutual_info_score
nmi=normalized_mutual_info_score(adata.obs['domain'], adata.obs['fine_annot_type'])
print('normalized mutual info score = %.5f' % nmi)
