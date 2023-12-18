library(Matrix)
library(SummarizedExperiment)

processData <- function(rna_file_path, atac_file_path, output_directory) {
  # Read both SummarizedExperiments
  scRNA <- readRDS(rna_file_path)
  scATAC <- readRDS(atac_file_path)

  # Save the classes for classification
  write.csv((colData(scRNA)$seurat_clusters), file.path(output_directory, "scRNAclasses.csv"))
  
  # Process scRNA data
  scRNA <- t(assay(scRNA))
  variances <- apply(scRNA, 2, var)
  top_1000_genes_by_variance <- (order(variances, decreasing = TRUE))[1:1000]
  scRNA <- scRNA[, top_1000_genes_by_variance]
  write(colnames(scRNA), file = file.path(output_directory, "scRNAcols.txt"))
  write(rownames(scRNA), file = file.path(output_directory, "scRNArows.txt"))
  writeMM(scRNA, file = file.path(output_directory, "scRNAmat.txt"))

  # Process scATAC data
  scATAC <- t(assay(scATAC))
  col_sums <- colSums(scATAC)
  top_1k_sum_peaks <- (order(col_sums, decreasing = TRUE))[1:1000]
  scATAC <- scATAC[, top_1k_sum_peaks]
  print("Dimensions of the matrix after filtering: ") # Sanity check
  write(colnames(scATAC), file = file.path(output_directory, "scATACcols.txt"))
  write(rownames(scATAC), file = file.path(output_directory, "scATACrows.txt"))
  writeMM(scATAC, file = file.path(output_directory, "scATACmat.txt"))
}

processData("path_to_rna_file", "path_to_atac_file", "output_directory_path")

peak2genelinnks <- read.xlsx("path_to_p2g_link_file")
write.csv2(peak2genelinnks, file = "output_directory_path", row.names = FALSE)

#path_to_rna_file should be your path to scRNA matrix file
#path_to_atac_file should be your path to scATAC matrix file
#path_to_p2g_link_file should be path to p2g link file
#output_directory_path should be where the files should be saved
