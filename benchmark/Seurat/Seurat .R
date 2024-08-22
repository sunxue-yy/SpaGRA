library(tidyverse)
library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
library(reticulate)


dataname <- "151676"
dataroot <- "E:/DLPFC"
Brain <- Load10X_Spatial('E:/DLPFC/151676')
Brain <- SCTransform(Brain, assay = "Spatial", variable.features.n = 3000, verbose = FALSE)

ScaleData <- Brain@assays$SCT@scale.data
npcs=50
Brain1 = Brain
Brain = Brain1
num = 30


Brain = Brain1
Brain <- RunPCA(Brain, assay = "SCT",npcs = npcs, verbose = FALSE)

pc.num = 1:num
Brain <- FindNeighbors(Brain, reduction = "pca", dims = pc.num)
Brain <- FindClusters(Brain, verbose = F, resolution = 0.5)

#Brain <- FindClusters(Brain, verbose = FALSE)
Brain <- RunUMAP(Brain, reduction = "pca", dims = pc.num)


label <- Brain@meta.data$seurat_clusters
n = table(label)
n
test0 <- Brain@reductions$pca@cell.embeddings[,1:num]
label <- as.integer(label)





