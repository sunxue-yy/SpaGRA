library(SpatialPCA)
library(ggplot2)



rawcount <- as.matrix(read.csv("C:/Users/Administrator/Desktop/PCA/expression_matrix_breast.csv", row.names = 1))
location <- read.csv("C:/Users/Administrator/Desktop/PCA/location_breast.csv", row.names = 1)

#load("C:/Users/Administrator/Desktop/BreastTumor/Tumor_data.RData") 

print(dim(rawcount)) # The count matrix
print(dim(location)) # The location matri

rawcount <- t(rawcount)
rawcount_matrix <- as.matrix(rawcount)
location_matrix <- as.matrix(location)

print(dim(rawcount_matrix)) # The count matrix
print(dim(location_matrix)) # The location matri

ST = CreateSpatialPCAObject(counts=rawcount_matrix, location=location_matrix, project = "SpatialPCA",gene.type="spatial",sparkversion="spark", gene.number=3000,customGenelist=NULL,min.loctions = 20, min.features=20)


ST = SpatialPCA_buildKernel(ST, kerneltype="gaussian", bandwidthtype="SJ")
ST = SpatialPCA_EstimateLoading(ST,fast=FALSE,SpatialPCnum=20)
ST = SpatialPCA_SpatialPCs(ST, fast=FALSE)

clusterlabel= walktrap_clustering(20, ST@SpatialPCs,round(sqrt(dim(ST@location)[1])))
clusterlabel_refine=refine_cluster_10x(clusterlabel,ST@location,shape="square")


# set color
# cbp_spatialpca = c(  "mediumaquamarine", "chocolate1","dodgerblue",  "#F0E442","palegreen4","lightblue2","plum1")
cbp_spatialpca <- c(
  "mediumaquamarine", "chocolate1", "dodgerblue", "#F0E442", "palegreen4",
  "lightblue2", "plum1", "darkorange", "firebrick1", "steelblue", 
  "yellowgreen", "sandybrown", "tomato", "orchid", "gold", 
  "darkseagreen", "cornflowerblue", "mediumvioletred", "cyan", "deepskyblue"
)

# visualize the cluster
plot_cluster(legend="right",location=ST@location,clusterlabel_refine,pointsize=2,text_size=20 ,title_in=paste0("SpatialPCA"),color_in=cbp_spatialpca)


