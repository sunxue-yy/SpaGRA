# Relation-aware graph augmentation with geometric contrastive learning improves the domains identification from spatially resolved transcriptomics data  
## 1.Introduction  
SpaNCL is based on graph augmentation and geometric contrastive learning to capture the latent representations of spots or cells. SpaNCL can effectively integrates gene expression and spatial location information of SRT data.  

  For graph augmentation, SpaNCL employs spatial distance as prior knowledge and updates graph relationships with multi-head GATs. Then, SpaNCL utilizes geometric contrastive learning to enhance the discriminability of the latent embeddings. Furthermore, SpaNCL leverage these multi-view relationships to construct more rational negative samples, which can significantly increase the recognition capabilities of SpaNCL. The model is trained using contrastive loss, similarity constraint loss and ZINB loss. Ultimately, SpaNCL applies the learned embeddings to cluster spatial domains.   

  The workflow of SpaNCL is shown in the following diagram.  
  
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

