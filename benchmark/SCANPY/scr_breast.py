import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
from datetime import datetime
from sklearn import metrics


section_id = "V1_Breast_Cancer_Block_A_Section_1"
k = 20
print(section_id,k)
input_dir = os.path.join('Data', "V1_Breast_Cancer_Block_A_Section_1")
adata = sc.read_visium(path=input_dir, count_file=section_id+'_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.louvain(adata, resolution=0.9)
sc.tl.umap(adata)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.embedding(adata, basis="spatial", color="louvain", s=6, show=False, title='SCANPY')
plt.axis('off')

Ann_df = pd.read_csv("Data/V1_Breast_Cancer_Block_A_Section_1/metadata.tsv", sep="	", header=0, na_filter=False,
                     index_col=0)
Ann_df["scanpy"]=adata.obs['louvain']
Ann_df.to_csv("scanpy_" + section_id + ".csv")
Ann_df.dropna(inplace=True)

ari = metrics.adjusted_rand_score(Ann_df["fine_annot_type"], Ann_df["scanpy"])
print(ari)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color="louvain", title='SCANPY (ARI=%.2f)' % ari,
           save="scanpy")

from sklearn.metrics import normalized_mutual_info_score
nmi=normalized_mutual_info_score(Ann_df["fine_annot_type"], Ann_df["scanpy"])
print('normalized mutual info score = %.5f' % nmi)



