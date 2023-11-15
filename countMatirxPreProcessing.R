
library(Matrix)
library(SummarizedExperiment)


#read both SummarizedExperiments
scRNA <- readRDS("/projectnb/paxlab/agokhale/EC523/Project/Multiome_RNA_SCE.RDS")
scATAC <- readRDS("/projectnb/paxlab/agokhale/EC523/Project/Multiome_ATAC_SCE.RDS")

#save the classes for classification
write.csv((colData(scRNA)$seurat_clusters), "/projectnb/paxlab/agokhale/EC523/Project/scRNAclasses.csv")
  
#filter out top 1k genes by variance and save it as a matrix market file so it can then be read in python
scRNA <- t(assay(scRNA))
variances <- apply(scRNA, 2, var)
top_1000_genes_by_variance <- (order(variances, decreasing = TRUE))[1:1000]
scRNA <- scRNA[, top_1000_genes_by_variance]
write(colnames(scRNA), file = "/projectnb/paxlab/agokhale/EC523/Project/scRNAcols.txt")
write(rownames(scRNA), file = "/projectnb/paxlab/agokhale/EC523/Project/scRNArows.txt")
writeMM(scRNA, file = "/projectnb/paxlab/agokhale/EC523/Project/scRNAmat.txt")



#filter out top 1k genes by sum and save it as a matrix market file so it can then be read in python
scATAC <- t(assay(scATAC))
col_sums <- colSums(scATAC)
top_1k_sum_peaks <- (order(col_sums, decreasing = TRUE))[1:1000]
scATAC <- scATAC[, top_1k_sum_peaks]
print("Dimensions of the matrix after filtering: ") #should have 1000 columns but a sanity check
write(colnames(scATAC), file = "/projectnb/paxlab/agokhale/EC523/Project/scATACcols.txt")
write(rownames(scATAC), file = "/projectnb/paxlab/agokhale/EC523/Project/scATACrows.txt")
writeMM(scATAC, file = "/projectnb/paxlab/agokhale/EC523/Project/scATACmat.txt")








