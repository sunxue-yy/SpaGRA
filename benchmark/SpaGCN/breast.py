#%%

import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN_package.SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from sklearn import metrics
#In order to read in image data, we need to install some package. Here we recommend package "opencv"
#inatll opencv in python
#!pip3 install opencv-python
import cv2
#Read original 10x_h5 data and save it to h5ad
from scanpy import read_10x_h5

dataset="V1_Breast_Cancer_Block_A_Section_1"
k=20
print(dataset, k)
adata = read_10x_h5("data/{}/{}_filtered_feature_bc_matrix.h5".format(dataset,dataset))
spatial=pd.read_csv("data/{}/spatial/tissue_positions_list.csv".format(dataset),
                    sep=",",header=None,na_filter=False,index_col=0)
img=cv2.imread("data/{}/{}_image.tif".format(dataset,dataset))
Ann_df = pd.read_csv("data/{}/metadata.tsv".format(dataset), sep="	", header=0, na_filter=False,
                     index_col=0)

adata.obs["x1"]=spatial[1]
adata.obs["x2"]=spatial[2]
adata.obs["x3"]=spatial[3]
adata.obs["x4"]=spatial[4]
adata.obs["x5"]=spatial[5]
#Select captured samples
adata=adata[adata.obs["x1"]==1]
adata.var_names=[i.upper() for i in list(adata.var_names)]
adata.var["genename"]=adata.var.index.astype("str")

adata.obs["x_array"]=adata.obs["x2"]
adata.obs["y_array"]=adata.obs["x3"]
adata.obs["x_pixel"]=adata.obs["x4"]
adata.obs["y_pixel"]=adata.obs["x5"]
x_array=adata.obs["x_array"].tolist()
y_array=adata.obs["y_array"].tolist()
x_pixel=adata.obs["x_pixel"].tolist()
y_pixel=adata.obs["y_pixel"].tolist()

# # Test coordinates on the image
# img_new = img.copy()
# for i in range(len(x_pixel)):
#     x = x_pixel[i]
#     y = y_pixel[i]
#     img_new[int(x - 20):int(x + 20), int(y - 20):int(y + 20), :] = 0
#
# cv2.imwrite("data/spatialLIBD/"+dataset+'/map.jpg', img_new)

# Calculate adjacent matrix
s = 1
b = 49
adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s,
                               histology=True)

adata.var_names_make_unique()
spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
spg.prefilter_specialgenes(adata)
# Normalize and take log for UMI
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

p = 0.5
# Find the l value given p
l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

# Set seed
r_seed = t_seed = n_seed = 100
# Seaech for suitable resolution
res = spg.search_res(adata, adj, l, k, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20,
                     r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
clf = spg.SpaGCN()
clf.set_l(l)
# Set seed
random.seed(r_seed)
torch.manual_seed(t_seed)
np.random.seed(n_seed)
# Run
clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
y_pred, prob = clf.predict()
adata.obs["pred"] = y_pred
adata.obs["pred"] = adata.obs["pred"].astype('category')
# Do cluster refinement(optional)
# shape="hexagon" for Visium data, "square" for ST data.
adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
                          shape="hexagon")
adata.obs["refined_pred"] = refined_pred
adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')

Ann_df["spagcn"] = adata.obs["pred"]
Ann_df["spagcn-r"] = adata.obs["refined_pred"]
# Ann_df.dropna(inplace=True)

ari = metrics.adjusted_rand_score(Ann_df["fine_annot_type"], Ann_df["spagcn"])
ari2 = metrics.adjusted_rand_score(Ann_df["fine_annot_type"], Ann_df["spagcn-r"])
Ann_df.to_csv("out/spagcn1_" + dataset + ".csv")
print(ari,ari2)

# sc.pp.neighbors(adata, use_rep='SpaNCL')
# sc.tl.umap(adata)
# plt.rcParams["figure.figsize"] = (3, 3)
# sc.pl.umap(adata, color=["SpaNCL", 'Ground Truth'], title=['SpaNCL (ARI=%.2f)' % ARI, 'Ground Truth'])
#
# plt.rcParams["figure.figsize"] = (3, 3)
# sc.pl.spatial(adata, color=["spagcn-r", 'Ground Truth'], title=['spagcn- (ARI=%.2f)' % ari, 'Ground Truth'],save="breastvancer")
#
#
from sklearn.metrics import normalized_mutual_info_score
nmi=normalized_mutual_info_score(Ann_df["fine_annot_type"], Ann_df["spagcn"])
nmi2=normalized_mutual_info_score(Ann_df["fine_annot_type"], Ann_df["spagcn-r"])
print(nmi,nmi2)

