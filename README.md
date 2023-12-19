Integrating ATAC-seq and RNA-seq for Cell Type Classification in Single-Cell Data


Ameya Gokhale, Idris Seidu, Jasper Huang, Oscar Zhang 
EC 523 FALL 2023 Project

Project goals:
1. Reimplement the sigGCN method as developed by [1] and extend it to in include scATAC-seq data. Analyze the effects of including scATAC-seq data on both reconstruction and overall classification accuracy. 
2. Randomize the networks to guage the significance of the specific gene-gene and gene-peak interactions to the model
3. Determine which peaks/genes are contributing the most to the reconstruction


STEP 1: get the data at the following links and save to your local directory 

https://atrev.s3.amazonaws.com/brainchromatin/Multiome_RNA_SCE.RDS (scrna data) 
https://atrev.s3.amazonaws.com/brainchromatin/Multiome_ATAC_SCE.RDS (scATAC matrix)
https://atrev.s3.amazonaws.com/brainchromatin/multiome_cell_metadata.txt (cell metadata)
https://atrev.s3.amazonaws.com/brainchromatin/multiome_cluster_names.txt (biological names of each cluster) 
https://www.cell.com/cms/10.1016/j.cell.2021.07.039/attachment/ff475640-7b35-4751-b7c6-f5332dc2a38e/mmc2.xlsx (peak to gene linkages file)



STEP 2:

To get the scRNA-seq, scATAC-matrices, cell classes and peak2gene linkages run the following command in terminal but with your paths for data saved in step 1. This will save the relevant data to your output directory.

Rscript process_data.R "/path/to/scRNA.rds" "/path/to/scATAC.rds" "/path/to/peak2gene.xlsx" "/path/to/output/"

Functions/Outputs of Rscript proces_data.R

1. Read in the scRNA, scATAC, summarized expriment files
2. filter peaks for top 20k peaks save both matrices in a way that they are readable in R
3. save cell labels for classification downstream

STEP 3: 

run the projscript.py file on a compute cluster. On the BU cluster the extra qsub options used was (-pe omp 16 for 16 cores) along with 1 V100 GPU (make sure you use this or you will run into memory issues as the matrices are large)
Note that the projscript.py file requries the user to manually code in the paths to some files in the first few lines. 


Functions/Output of projscrip.py:

1. Read in -omics dataset, peaks2genelinkages, HINT lab protein-protein interactions
2. construct adjacency matrices - gene-gene, gene-peak , gene-gene (randomized), gene-peak (randomized)
3. Run baselind NN and RF
4. run sigGCN and our extended model:
     a. reconstruction: cell-wise split validation, gene-mask validation for all four data configurations (gene-gene and scRNA, gene-peak and scRNA and scATAC, randomized gene-gene and scRNA, randomized gene-peak and         scRNA+scATAC
     b. classification: sigGCN, extended sigGCN classification


Sources: 
1. Wang, T., Bai, J. & Nabavi, S. Single-cell classification using graph convolutional networks. BMC Bioinformatics 22, 364 (2021).
