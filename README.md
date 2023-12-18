

To get the scRNA-seq, scATAC-matrices, cell classes and peak2gene linkages run the following command in terminal but with your paths. This will save the relevant data to your output directory.

Rscript process_data.R "/path/to/scRNA.rds" "/path/to/scATAC.rds" "/path/to/peak2gene.xlsx" "/path/to/output/"
