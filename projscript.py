from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import time
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, global_max_pool, TopKPooling
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy import io
from scipy.spatial import distance as sp
from scipy.spatial.distance import pdist, jaccard
from sklearn.metrics.pairwise import cosine_similarity
import mygene
from tqdm import tqdm
from model_data_preparation import get_df, process_scRNA_data, get_data_NN, generate_adjacency_matrix, jaccard_similarity_matrix_pytorch, maslov_sneppen_rewire_one_swap_per_iter, preproces_joined_data
from baseline_models import NN, train_NN, test_NN, random_forest
from model_funcs import pad_test_data,train_GCN_autoencoder, calculate_accuracy, collate_fn_gene_only, prep_gene_autoencoder, custom_collate_fn_mask, prepare_data_masing, test_model_full_reconstruction,test_model_on_masked_genes, prepare_data_for_extended_model, train_model_extended_reconstruction, CombinedDataset, collate_fn_joint_model, prepare_data_loader_combined, train_combined_model, test_combined_model


if __name__ == "__main__":

    #mat, row, cols names assuming your working directory is where all these are saved
    scRNA_paths = ['scRNAmat.txt', 'scRNArows.txt', 'scRNAcols.txt']
    scATAC_paths = ['atac_sparsematrix.txt', 'row_atac.txt', 'col_atac.txt']
    protein_interaction_file = ['HomoSapiens_cocomp_hq.txt']
    
    #create scRNA_seq and scATAC_seq data
    scRNA = get_df(scRNA_paths)
    scATAC = get_df(scATAC_paths)
    
    #get the metadata file (that contains the cell labels) with info for each cell, drop most of the columns that arent relevenat. 
    cell_meta_data = pd.read_csv('multiome_cell_metadata.txt', sep = "\t")
    cell_meta_data = cell_meta_data.loc[:, ['Cell.ID', 'seurat_clusters']]
    
    #make sure all three dataframes (scRNA-seq rows, scATAC-seq rowws and cell label file) have the same ordering of cells
    #remember that the dataframe indices of the omics data are cells
    cell_meta_data.set_index('Cell.ID', inplace = True)
    scRNA = scRNA.reindex(scATAC.index)
    
    
    #replace the seurat clusters with the actual biological classes 
    cell_meta_data = cell_meta_data.reindex(scATAC.index)
    cluster_names = pd.read_csv('multiome_cluster_names.txt', sep = "\t")
    cluster_maps = {k: v for k, v in zip(cluster_names['Cluster.ID'], cluster_names['Cluster.Name'])}
    cell_meta_data['seurat_clusters'] = cell_meta_data['seurat_clusters'].map(cluster_maps)
    labels = list(cell_meta_data['seurat_clusters'])

    gene_names = list(scRNA.columns)
    peak_names - list(scATAC.columns)

    scRNA = process_scRNA_data(scRNA)

    #drop duplicated peaks and get the names of the peaks which are saved in a pkl list
    scATAC = scATAC.T.drop_duplicates().T
    
  # Load peak2genes data and keep peaks and genes in the file
    peak2genes = pd.read_csv('peak2gene.csv', sep=";")
    peak2genes.columns = ['peak name', 'gene name', 'correlation']
    peak2genes = peak2genes[(peak2genes['gene name'].isin(gene_names)) & (peak2genes['peak name'].isin(scATAC.columns))]
    peak2genes['correlation'] = peak2genes['correlation'].astype(float)

    #for baseline RF and simple neural network
    train, val, test = get_data_NN("scRNA", scRNA, scATAC, labels)
    net = NN()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net = train_NN(net, optimizer, device, train, criterion, num_epochs = 75)
    predicted_labels, true_labels = test_NN(net, test)
  
    X_train, X_test, y_train, y_test = train_test_split(scRNA, labels, test_size=0.2, random_state=42)
    random_forest(X_train, y_train, X_test, y_test)


    #get protein-protein interaction files
    known_interactions = pd.read_csv(protein_interaction_file, sep = "\t")
    known_interactions = known_interactions[known_interactions['Gene_A'].isin(gene_names) & known_interactions['Gene_B'].isin(gene_names)]

    #create adjacency_matrices 

    #gene-gene
    adj_matrix = generate_adjacency_matrix(gene_names)

    #gene-peak
    peak_by_cell_binary = (scATAC > 0).astype(int)
    # create jaccard similarity matrix
    jaccard_matrix = jaccard_similarity_matrix_pytorch(peak_by_cell_binary, chunk_size=100)
    num_genes = len(gene_names)  # Replace with your gene count
    num_peaks = jaccard_matrix.shape[0]
    # following code is for generating the peak-gene edges based on the prior peak2genes data
    peak_to_idx = {peak: idx + len(gene_names) for idx, peak in enumerate(unique_peaks)}
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    total_nodes = len(gene_names) + len(unique_peaks)
    new_adj_matrix = np.zeros((total_nodes, total_nodes))
    gene_gene_size = len(gene_names)
    new_adj_matrix[:gene_gene_size, :gene_gene_size] = adj_matrix
    for _, row in tqdm(peak2genes.iterrows()):
        peak_idx = peak_to_idx[row['peak name']]
        gene_idx = gene_to_idx[row['gene name']]
        new_adj_matrix[peak_idx, gene_idx] = 1  
        new_adj_matrix[gene_idx, peak_idx] = 1  
    # Insert Jaccard similarity matrix into the new adjacency matrix
    start_index = num_genes  
    end_index = start_index + num_peaks  
    new_adj_matrix[start_index:end_index, start_index:end_index] = jaccard_matrix

    #get combined data: 
    sc_joined, loaded_adj_matrix = preprocess_joined_data(scRNA, scATAC, loaded_adj_matrix, num_peaks_to_keep=15000)
    sc_joined_train, sc_joined_test = prepare_data_loader_combined(sc_joined, labels, loaded_adj_matrix, test_size=0.2, random_state=42, batch_size=32, shuffle=True)

    #create rewired matrices
    rewired_base_matrix = maslov_sneppen_rewire_one_swap_per_iter(adj_matrix)
    rewired_extended_matrix = maslov_sneppen_rewire_one_swap_per_iter(new_adj_matrix)

    #cell wies autoencoder reconstruction:
  
    #gene gene data (non rewired)
    train_loader_gene_reconstruction, test_data_data_gene_reconstruction = prep_gene_autoencoder_data(scRNA, adj_matrix, batch_size=32, test_size=0.2, random_state=42)
    num_genes = int(0.8*len(scRNA.index)) #Number of genes
    hidden_dim = 64

    model_gene_cell_wise = GCNAutoencoder_gene_only(num_genes, hidden_dim)
    model_gene_cell_wise = train_GCN_autoencoder(model, loader, num_epochs=100, learning_rate=0.001, weight_decay=0.01)

    #get reconstruction accuracy
    model_gene_cell_wise.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        reconstructed_test = model_gene_cell_wise(test_data_data_gene_reconstruction)
        test_loss = criterion(reconstructed_test, test_data_data_gene_reconstruction.x)
        print(f"Test Loss: {test_loss}")

    #gene gene data (rewired)
    num_iterations = 1000  # Set the number of iterations
    rewired_matrix = maslov_sneppen_rewire_one_swap_per_iter(adj_matrix, num_iterations)
    
    
    train_loader_gene_reconstruction, test_data_data_gene_reconstruction = prep_gene_autoencoder_data(scRNA, rewired_matrix, batch_size=32, test_size=0.2, random_state=42)
    num_genes = int(0.8*len(scRNA.index)) #Number of genes
    hidden_dim = 64
    
    model = GCNAutoencoder(num_genes, hidden_dim)
    model_gene_cell_wise_rewired = train_GCN_autoencoder(model, loader, num_epochs=100, learning_rate=0.001, weight_decay=0.01)
    
    model_gene_cell_wise.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        reconstructed_test = model_gene_cell_wise_rewired(test_data_data_gene_reconstruction)
        test_loss = criterion(reconstructed_test, test_data_data_gene_reconstruction.x)
        print(f"Test Loss: {test_loss}")

    
    #gene_peak_data (extended) note this code is the same as above and just has the matrix changed
    train_loader_gene_reconstruction, test_data_data_gene_reconstruction = prep_gene_autoencoder_data(sc_joined, adj_matrix, batch_size=32, test_size=0.2, random_state=42)
    num_genes = int(0.8*len(scRNA.index)) #Number of genes
    hidden_dim = 64

    model_gene_cell_wise = GCNAutoencoder_gene_only(num_genes, hidden_dim)
    model_gene_cell_wise = train_GCN_autoencoder(model, loader, num_epochs=100, learning_rate=0.001, weight_decay=0.01)

    #get reconstruction accuracy
    model_gene_cell_wise.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        reconstructed_test = model_gene_cell_wise(test_data_data_gene_reconstruction)
        test_loss = criterion(reconstructed_test, test_data_data_gene_reconstruction.x)
        print(f"Test Loss: {test_loss}")


    
    
    train_loader_gene_reconstruction, test_data_data_gene_reconstruction = prep_gene_autoencoder_data(sc_joined, rewired_matrix, batch_size=32, test_size=0.2, random_state=42)
    num_genes = int(0.8*len(scRNA.index)) #Number of genes
    hidden_dim = 64
    
    model = GCNAutoencoder(num_genes, hidden_dim)
    model_gene_cell_wise_rewired = train_GCN_autoencoder(model, loader, num_epochs=100, learning_rate=0.001, weight_decay=0.01)
    
    model_gene_cell_wise.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        reconstructed_test = model_gene_cell_wise_rewired(test_data_data_gene_reconstruction)
        test_loss = criterion(reconstructed_test, test_data_data_gene_reconstruction.x)
        print(f"Test Loss: {test_loss}")

    #creating saliency map
    data.x.requires_grad_(True)
    model.eval()
    reconstructed_output = model(data)
    
    # Compute loss
    test_loss = criterion(reconstructed_output, data.x)
    
    # Backpropagate to compute gradients
    model.zero_grad()
    test_loss.backward()
    
    # Aggregate gradients row-wise
    row_gradients = torch.mean(data.x.grad, dim=1)  # Summing across columns
    peak_gradients = row_gradients[-5000:]
    gene_gradients = row_gradients[:-5000]
    
    sns.set(style="whitegrid")
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(np.log(sorted_peak_gradients), bins=50, kde=True, color="skyblue")
    
    # Labeling
    plt.xlabel('Gradient Values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Histogram of Gradient Values for Peaks', fontsize=14)
    
    # Save the plot to a file
    plt.savefig('peak_gradients.png', format='png', bbox_inches='tight', dpi=300)
    
    plt.show()
  
    #masking reconstruction (gene only) for both normal and rewired: 
    for i in [adj_matrix, rewired_base_matrix]
      train_loader, test_loader = def prepare_data_for_extended_model(scRNA, adj_matrix, fraction_to_mask, batch_size=32, test_size=0.2, random_state=42)
  
      test_loss_full = test_model_full_reconstruction(model, test_loader, criterion)
      print(f"Average Test Loss on Full Gene Expression Data: {test_loss_full}")
        
      test_loss_masked = test_model_on_masked_genes(model, test_loader, criterion, masked_genes_indices)
      print(f"Average Test Loss on Masked Genes: {test_loss_masked}")
  
      #masking reconstruction (gene-peak only): 
      train_loader, test_loader = prepare_data_masking(scRNA, 0.3)
  
      test_loss_full = test_model_full_reconstruction(model, test_loader, criterion)
      print(f"Average Test Loss on Full Gene Expression Data: {test_loss_full}")
        
      test_loss_masked = test_model_on_masked_genes(model, test_loader, criterion, masked_genes_indices)
      print(f"Average Test Loss on Masked Genes: {test_loss_masked}") 

    #masking reconstruction (gene-peakonly) for both normal and rewired: 
    for i in [loaded_adj_matrix, rewired_extended_matrix]
      train_loader, test_loader = def prepare_data_for_extended_model(sc_joined, adj_matrix, fraction_to_mask, batch_size=32, test_size=0.2, random_state=42)
  
      test_loss_full = test_model_full_reconstruction(model, test_loader, criterion)
      print(f"Average Test Loss on Full Gene Expression Data: {test_loss_full}")
        
      test_loss_masked = test_model_on_masked_genes(model, test_loader, criterion, masked_genes_indices)
      print(f"Average Test Loss on Masked Genes: {test_loss_masked}")
  
      #masking reconstruction (gene-peak only): 
      train_loader, test_loader = prepare_data_masking(scRNA, 0.3)
  
      test_loss_full = test_model_full_reconstruction(model, test_loader, criterion)
      print(f"Average Test Loss on Full Gene Expression Data: {test_loss_full}")
        
      test_loss_masked = test_model_on_masked_genes(model, test_loader, criterion, masked_genes_indices)
      print(f"Average Test Loss on Masked Genes: {test_loss_masked}") 



    #combeind model classification 
    #basic sigGCN and extended: 
      for (i,j) in [(sc_rna, adj_matrix), (sc_joined, new_adj_matrix)]:
        train_sigcn, test_sigcn =  prepare_data_loader_combined(sc_rna, labels, adj_matrix, test_size=0.2, random_state=42, batch_size=32, shuffle=True)
        model = CombinedModel(sc_rna.shape[1], 32, 14)  # Adjust parameters as needed
        train_combined_model(model, loader, epochs=75, learning_rate=0.001, weight_decay=0.01, device='cuda')
        test_combined_model(model, test_loader, device='cuda')                    



    
   
    
    
   
