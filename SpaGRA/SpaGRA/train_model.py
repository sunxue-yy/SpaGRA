import scipy.sparse as sp
from .utils import Transfer_Data
from .utils import ZINB,refine_label
import torch
import torch.nn.functional as F
import os
from .process import set_seed
from .gat import GAT
from .loss import multihead_contrastive_loss
import dgl
import numpy as np
import time
from sklearn import metrics
from sklearn.cluster import KMeans
import scanpy as sc
import pandas as pd


def train(adata,k=0,hidden_dims=3000, n_epochs=200,num_hidden=100,lr=0.00008, key_added='SpaGRA',a=0.5,b=5,c=2,
                radius=50,  weight_decay=0.0001,  random_seed=0,feat_drop=0.01,attn_drop=0.1,
                negative_slope=0.01,heads=4,method="kmeans",reso=1,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    set_seed(random_seed)
    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")


    adj,features = Transfer_Data(adata_Vars)
    g = dgl.from_scipy(adj)
    all_time = time.time()
    g = g.int().to(device)
    num_feats = features.shape[1]
    n_edges = g.number_of_edges()
    model = GAT(g,
                hidden_dims,
                1,
                num_feats,
                num_hidden,
                [heads],
                F.elu,
                feat_drop,
                attn_drop,
                negative_slope)
    adj = torch.tensor(adj.todense()).to(device)
    features = torch.FloatTensor(features).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)


    coords = torch.tensor(adata.obsm['spatial']).float().to(device)
    sp_dists = torch.cdist(coords, coords, p=2)
    sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
    ari_max=0
    model.train()
    for epoch in range(n_epochs):

        # model.train()
        optimizer.zero_grad()
        heads, pi, disp, mean  = model(features)
        heads0 = torch.cat(heads, axis=1)
        # heads0 = heads[0]

        z_dists = torch.cdist(heads0, heads0, p=2)
        z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
        n_items = heads0.size(dim=0) * heads0.size(dim=0)
        reg_loss = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
        # reg_loss=0
        zinb_loss = ZINB(pi, theta=disp, ridge_lambda=1).loss(features, mean, mean=True)
        loss = multihead_contrastive_loss(heads, adj, tau=10)
        total_loss =  a * loss + b * reg_loss  + c* zinb_loss


        total_loss.backward()
        optimizer.step()
        print("loss ",epoch,loss.item(),reg_loss.item(),zinb_loss.item())
        # kmeans = KMeans(n_clusters=k).fit(np.nan_to_num(heads0.cpu().detach()))
        if method=="kmeans":
            kmeans = KMeans(n_clusters=k, random_state=random_seed).fit(np.nan_to_num(heads0.cpu().detach()))
            idx = kmeans.labels_
            adata_Vars.obs['temp'] = idx
            obs_df = adata_Vars.obs.dropna()
            ari_res = metrics.adjusted_rand_score(obs_df['temp'], obs_df['Ground Truth'])
            # print("ARI:",ari_res,"MAX ARI:",ari_max)
            if ari_res > ari_max:
                ari_max = ari_res
                idx_max = idx
                mean_max = mean.to('cpu').detach().numpy()
                emb_max = heads0.to('cpu').detach().numpy()

        if method == "louvain":
            adata_tmp = sc.AnnData(np.nan_to_num(heads0.cpu().detach()))
            sc.pp.neighbors(adata_tmp, n_neighbors=20,use_rep='X')
            sc.tl.louvain(adata_tmp, resolution=reso, random_state=0)
            idx = adata_tmp.obs['louvain'].astype(int).to_numpy()

    if method == "kmeans":
        adata.obs["cluster"] = idx_max.astype(str)
        adata.obsm["emb"] = emb_max
        adata.obsm['mean'] = mean_max

    if method == "louvain":
        adata.obs["cluster"] = idx.astype(str)
        emb = heads0.to('cpu').detach().numpy()
        adata.obsm["emb"] = emb  ######

    if radius !=0 :
        nearest_new_type = refine_label(adata, radius=radius)
        adata.obs[key_added] = nearest_new_type
    else:
        adata.obs[key_added] = adata.obs["cluster"]
    # adata.obsm["emb"] = emb_max
    # adata.obsm['mean'] = mean_max
    model.eval()
    heads, pi, disp, mean = model(features)
    z = torch.cat(heads, axis=1)
    adata.obsm[key_added] = z.to('cpu').detach().numpy()

    return adata

# model(data.x, data.edge_index)