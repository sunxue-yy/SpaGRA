import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
# import sklearn
import STAGATE_pyG as STAGATE
os.environ['R_HOME'] = '/home/dell/anaconda3/envs/stagate/lib/R'
# os.environ['R_USER'] = 'D:\ProgramData\Anaconda3\Lib\site-packages/rpy2'


section_id="V1_Breast_Cancer_Block_A_Section_1"
k=20
print(section_id,k)
input_dir = os.path.join('Data', section_id)
adata = sc.read_visium(path=input_dir, count_file=section_id+'_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


Ann_df =  pd.read_csv("Data/V1_Breast_Cancer_Block_A_Section_1/metadata.tsv", sep="	", header=0, na_filter=False,
                          index_col=0)
adata.obs['fine_annot_type'] = Ann_df.loc[adata.obs_names, 'fine_annot_type']

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color=["fine_annot_type"])

# STAGATE.Cal_Spatial_Net(adata, k_cutoff=6,model='KNN')
STAGATE.Cal_Spatial_Net(adata, rad_cutoff=400)
STAGATE.Stats_Spatial_Net(adata)

adata = STAGATE.train_STAGATE(adata)

sc.pp.neighbors(adata, use_rep='STAGATE')
sc.tl.umap(adata)
adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=k)
tt=adata.obs
tt.to_csv("Data/ttt-stagate_" + section_id + ".csv")

obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['mclust'], obs_df['fine_annot_type'])
print('Adjusted rand index = %.5f' %ARI)

obs_df.to_csv("Data/stagate_" + section_id + ".csv")

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color="mclust", title='STAGATE (ARI=%.2f)' % ARI,
           save=section_id)


from sklearn.metrics import normalized_mutual_info_score
nmi=normalized_mutual_info_score(obs_df['mclust'], obs_df['fine_annot_type'])
print('normalized mutual info score = %.5f' % nmi)
