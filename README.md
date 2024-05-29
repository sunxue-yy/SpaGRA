# Relation-aware graph augmentation with geometric contrastive learning improves the domains identification from spatially resolved transcriptomics data  
## 1.Introduction  
SpaNCL is based on graph augmentation and geometric contrastive learning to capture the latent representations of spots or cells. SpaNCL can effectively integrates gene expression and spatial location information of SRT data.  

  For graph augmentation, SpaNCL employs spatial distance as prior knowledge and updates graph relationships with multi-head GATs. Then, SpaNCL utilizes geometric contrastive learning to enhance the discriminability of the latent embeddings. Furthermore, SpaNCL leverage these multi-view relationships to construct more rational negative samples, which can significantly increase the recognition capabilities of SpaNCL. The model is trained using contrastive loss, similarity constraint loss and ZINB loss. Ultimately, SpaNCL applies the learned embeddings to cluster spatial domains.   

  The workflow of SpaNCL is shown in the following diagram.  
  
