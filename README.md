# Relation-aware graph augmentation with geometric contrastive learning improves the domains identification from spatially resolved transcriptomics data  
## 1.Introduction  
SpaNCL is based on graph augmentation and geometric contrastive learning to capture the latent representations of spots or cells. SpaNCL can effectively integrates gene expression and spatial location information of SRT data.  

  For graph augmentation, SpaNCL employs spatial distance as prior knowledge and updates graph relationships with multi-head GATs. Then, SpaNCL utilizes geometric contrastive learning to enhance the discriminability of the latent embeddings. Furthermore, SpaNCL leverage these multi-view relationships to construct more rational negative samples, which can significantly increase the recognition capabilities of SpaNCL. The model is trained using contrastive loss, similarity constraint loss and ZINB loss. Ultimately, SpaNCL applies the learned embeddings to cluster spatial domains.   

  The workflow of SpaNCL is shown in the following diagram.      
  
  ![image](https://github.com/sunxue-yy/SpaNCL/blob/main/workflow.png "workflow of SpaNCL")
  
## 2.Requirements  
numpy==1.21.5  
torch==1.11.0  
pandas==1.3.5  
numba==0.55.1  
scanpy==1.9.1  
scikit-learn==1.0.2  
scipy==1.7.3  
anndata==0.8.0  
matplotlib==3.5.2

## 3.Datasets
All datasets used in this paper are publicly available. Users can download them from the links below.  
### Human breast cancer  
https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1  
### Mouse hypothalamus  
https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248  
### Mouse primary visual area  
https://spacetx.github.io/data.html  
### Mouse embryo  
https://db.cngb.org/stomics/mosta/  
Processed datasets are also available at SODB (https://gene.ai.tencent.com/SpatialOmics/) and can be loaded by PySODB (https://protocols-pysodb.readthedocs.io/en/latest/).  
### Visium HD  
https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-intestine  
https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc   

## 4.Usage  
### We provided some demos to demonstrate usage of SpaNCL.    
### Human breast cancer
```python  
adata = sc.read_visium("Data/V1_Breast_Cancer_Block_A_Section_1",
                count_file="V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
Ann_df = pd.read_csv("Data/V1_Breast_Cancer_Block_A_Section_1/metadata.tsv", sep="	", header=0, na_filter=False,
                     index_col=0)
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'fine_annot_type']
Cal_Spatial_Net(adata, rad_cutoff=400)
adata = train_model.train(adata,k=20,n_epochs=200)
obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['SpaNCL'], obs_df['Ground Truth'])
print('Adjusted rand index = %.2f' % ARI)

sc.pp.neighbors(adata, use_rep='SpaNCL')
sc.tl.umap(adata)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata, color=["SpaNCL", 'Ground Truth'], title=['SpaNCL (ARI=%.2f)' % ARI, 'Ground Truth'],save="SpaNCL_umap")

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color=["SpaNCL", 'Ground Truth'], title=['SpaNCL (ARI=%.2f)' % ARI, 'Ground Truth'],save="SpaNCL")  
```
### Mouse embryo  
```python
adata = sc.read("Data/embroy/95.h5ad")
adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata.obs['Ground Truth'] = adata.obs['annotation']

Cal_Spatial_Net(adata, rad_cutoff=10)
adata = train_model.train(adata,k=12,hidden_dims=500, n_epochs=200,num_hidden=400,lr=0.00005, key_added='SpaNCL',a=2,b=1,c=1,
                radius=20,  weight_decay=0.0001,  random_seed=0,feat_drop=0.01,attn_drop=0.02,
                negative_slope=0.02,heads=4,)

Ann_df = adata.obs.dropna()
ARI = adjusted_rand_score(Ann_df['SpaNCL'], Ann_df["annotation"])
print(ARI)

coor = pd.DataFrame(adata.obsm['spatial'])
coor.index = adata.obs.index
coor.columns = ['imagerow', 'imagecol']
adata.obs["y_pixel"]=coor['imagerow']
adata.obs["x_pixel"]=coor['imagecol']

domains='SpaNCL'
title = 'SpaNCL (ARI=%.2f)' % ARI
ax = sc.pl.scatter(adata, alpha=1, x="y_pixel", y="x_pixel", color=domains, legend_fontsize=18, show=False,
                   size=100000 / adata.shape[0])

ax.set_title(title, fontsize=23)
ax.set_aspect('equal', 'box')
ax.set_xticks([])
ax.set_yticks([])
ax.axes.invert_yaxis()
plt.savefig("Data/SpaNCL_embroy.pdf")
plt.close()
```
### Visium HD  
```python
adata = sc.read("Data/HD/HD2.h5ad")

adata.var_names_make_unique()
prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
Cal_Spatial_Net(adata, rad_cutoff=150)

adata = train_model.train(adata,k=7,hidden_dims=1000, n_epochs=200,num_hidden=600,lr=0.00005, key_added='SpaNCL',a=2,b=1,c=1,
                radius=0,  weight_decay=0.00001,  random_seed=0,feat_drop=0.02,attn_drop=0.01,
                negative_slope=0.02,heads=4,)

coor = pd.DataFrame(adata.obsm['spatial'])
coor.index = adata.obs.index
coor.columns = ['imagerow', 'imagecol']
adata.obs["x_pixel"]=coor['imagerow']
adata.obs["y_pixel"]=coor['imagecol']

domains='SpaNCL'
title = 'SpaNCL'
ax = sc.pl.scatter(adata, alpha=1, x="x_pixel", y="y_pixel", color=domains, legend_fontsize=18, show=False,
                   size=100000 / adata.shape[0])
ax.set_title(title, fontsize=23)
ax.set_aspect('equal', 'box')
ax.set_xticks([])
ax.set_yticks([])
ax.axes.invert_yaxis()
plt.savefig("HD2.pdf")
plt.close()
```
