class GCNAutoencoder_gene_only(nn.Module):
    #code for basic GCN autoencoder for only gene-gene data sigGCN
    def __init__(self, num_features, hidden_dim):
        super(GCNAutoencoder_gene_only, self).__init__()
        # encoder
        self.gcn = GCNConv(num_features, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # Decoder
        self.fc2 = nn.Linear(hidden_dim, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # encoder
        x = F.relu(self.gcn(x, edge_index))
        x = self.fc1(x)
        # decoder
        x = self.fc2(x)
        return x

class GCNAutoencoder_genes_and_peaks(nn.Module):
  #code for extended gcn with gene and peak network
    def __init__(self, num_features, hidden_dim):
        super(GCNAutoencoder_genes_and_peaks, self).__init__()
        # Shared encoder for both genes and peaks
        self.gcn = GCNConv(num_features, hidden_dim)

        # Separate fully connected layers for genes and peaks
        self.fc_genes = nn.Linear(hidden_dim, hidden_dim)
        self.fc_peaks = nn.Linear(hidden_dim, hidden_dim)

        # Decoder
        self.fc_decoder = nn.Linear(hidden_dim , num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply GCN to all features
        encoded = F.relu(self.gcn(x, edge_index))

        # split the parameters
        genes_encoded = encoded[:17675]  # First 17675 rows are genes
        peaks_encoded = encoded[17675:]  # Remaining are peaks

        # Apply separate linear transformations
        genes_transformed = self.fc_genes(genes_encoded)
        peaks_transformed = self.fc_peaks(peaks_encoded)

        # Concatenate the transformed features
        combined = torch.cat((genes_transformed, peaks_transformed), dim=0)

        # Decoder
        decoded = self.fc_decoder(combined)

        return  encoded, decoded
  

class NN(nn.Module):
  #note this is just the same as the basline method its copied without the fc3 as the siggcn model only cares about the feature space
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear( 17675, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 14)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
      
class sigGCN(nn.Module):
    def __init__(self, num_features_gcn, hidden_dim, num_classes):
        super(sigGCN, self).__init__()
        self.nn = NN()  
        self.gcn_autoencoder = GCNAutoencoder(num_features_gcn, hidden_dim)
        nn_output_size =num_classes
        self.classifier = nn.Linear(17707, num_classes)

    def forward(self, x_nn, data_gcn):
        # Forward pass through both networks
        x1 = self.nn(x_nn)
        reconstructed_x, x2 = self.gcn_autoencoder(data_gcn)

        # Concatenate features
        x2 = x2.T
        x2_weighted = x2

        x = torch.cat((x1, x2_weighted), dim=1)
        out = self.classifier(x)
        return F.log_softmax(out, dim=1), reconstructed_x
        return out
      
class extended_sigGCN(nn.Module):
    def __init__(self, num_features_gcn, hidden_dim, num_classes):
        super(CombinedModel_gens_and_peaks, self).__init__()
        self.nn = NN()  
        self.gcn_autoencoder = GCNAutoencoder_genes_and_peaks(num_features_gcn, hidden_dim)
        nn_output_size = num_classes  # Replace with the actual output size of your NN
        self.classifier = nn.Linear(num_features_gcn+hidden_dim, num_classes)

    def forward(self, x_nn, data_gcn):
        # Forward pass through both networks
        x1 = self.nn(x_nn)
        reconstructed_x, x2 = self.gcn_autoencoder(data_gcn)

        # Concatenate features
      
        x2 = x2.T
        x2_weighted = x2
        x = torch.cat((x1, x2_weighted), dim=1)
        # Classification
        out = self.classifier(x)
        return F.log_softmax(out, dim=1), reconstructed_x
