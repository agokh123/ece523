#for all analyses:
def pad_test_data(test_df, train_df):
    """
    Pad test data with 0s to match dimensions of training data

    Args:
    - test_df (pd.DataFrame): DataFrame containing test data
    - train_df (pd.DataFrame): DataFrame containing training data

    Returns:
    - padded_test_df (pd.DataFrame): Padded test data with same dims as training data
    """
    # Transpose the dataframes to have genes as rows and cells as columns
    test_df_transposed = test_df.transpose()
    train_df_transposed = train_df.transpose()

    # Calculate the difference in the number of cells
    num_cells_diff = train_df_transposed.shape[1] - test_df_transposed.shape[1]

    # Pad with zeros if test data has fewer cells
    if num_cells_diff > 0:
        padding = pd.DataFrame(np.zeros((test_df_transposed.shape[0], num_cells_diff)),
                               index=test_df_transposed.index)
        padded_test_df = pd.concat([test_df_transposed, padding], axis=1)
    else:
        padded_test_df = test_df_transposed

    return padded_test_df.transpose()

def train_GCN_autoencoder(model, loader, num_epochs=800, learning_rate=0.001, weight_decay=0.01):
    """
    Train a GCN autoencoder model (for both gene-gene and gene-peak networks, and both cell-wise and masking split)

    Args:
    - model (GCNAutoencoder): The GCN autoencoder model to be trained
    - loader (DataLoader): DataLoader containing the training data
    - num_epochs (int): Number of training epochs (default is 800)
    - learning_rate (float): Learning rate for the optimizer (default is 0.001)
    - weight_decay (float): Weight decay for regularization (default is 0.01)
    - print_interval (int): Interval for printing training loss (default is 50)

    Returns:
    - model (GCNAutoencoder): The trained GCN autoencoder model.
    - losses (list): List of average training losses for each epoch.
    """

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    losses = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0

        for batch in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.x)  # Assuming 'batch.x' contains the input data
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #used to make sure the model is converging
        avg_epoch_loss = epoch_loss / len(loader)
        losses.append(avg_epoch_loss)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: Average Training Loss = {avg_epoch_loss}")


    return model

def calculate_accuracy(output, labels):
   """
    Train a GCN autoencoder model (for both gene-gene and gene-peak networks, and both cell-wise and masking split)

    Args:
    - output: predicted output
    - labels: ground truth

    Returns:
    - accuracy: accuracy of model (porition of correctly predicted labels)
    """

    # Convert output probabilities to predicted class labels
    preds = output.argmax(dim=1, keepdim=True)
    correct = preds.eq(labels.view_as(preds)).sum().item()
    accuracy = correct / len(labels)
    return accuracy


#for GCN autoencoder (gene only)

def collate_fn_gene_only(batch):
    """
    Pytorch's collate function doesnt work for graph data so have to create custom one

    Args:
    - batch (list): Tuples of form (nn_data_item, gcn_data_item, label_item)

    Returns:
    - nn_data (torch.Tensor): nn_data items
    - gcn_data (Data object): gcn_data item in batch
    - labels (torch.Tensor): Batch label tensor
    """
    nn_data, gcn_data, labels = [], [], []
    for item in batch:
        nn_data_item, gcn_data_item, label_item = item
        nn_data.append(nn_data_item)
        # Assuming gcn_data_item is already a Data object; no need to append
        labels.append(torch.tensor(label_item, dtype=torch.long))

    nn_data = torch.stack(nn_data)
    labels = torch.stack(labels)

    # Assuming gcn_data is the same for all, so just use the first one from the batch
    gcn_data = batch[0][1]

    return nn_data, gcn_data, labels


def prep_gene_autoencoder_data(scRNA, adj_matrix, batch_size=32, test_size=0.2, random_state=42):
    """
    Prepare data for a Graph Convolutional Network (GCN) using PyTorch Geometric.

    Args:
    - scRNA (pd.DataFrame): DataFrame containing single-cell RNA-seq data
    - adj_matrix (numpy.ndarray): The adjacency matrix representing the graph
    - batch_size (int): Batch size for DataLoader (default is 32)
    - test_size (float): Fraction of the data to be used for testing (default is 0.2)
    - random_state (int): Random seed for reproducibility (default is 42)

    Returns:
    - loader (DataLoader): DataLoader for GCN training with the specified batch size.
    """

    # Split the data into training and test sets and pad test data for future use
    train_df, test_df = train_test_split(scRNA, test_size=test_size, random_state=random_state)

    # Convert data to tnesors and get edge index as adj matrix represenation
    gene_count_df = train_df.T
    gene_count_matrix = torch.tensor(gene_count_df.values, dtype=torch.float32)
    adjacency_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()

    # Create dataset for training data and make it into loader
    dataset = [Data(x=gene_count_matrix, edge_index=edge_index)]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    #create data for testing
    padded_test_df = pad_test_data(test_df, train_df)
    gene_count_df_test = padded_test_df.T
    gene_count_matrix_test = torch.tensor(gene_count_df_test.values, dtype=torch.float32)
    adjacency_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
    data_test = Data(x=gene_count_matrix_test, edge_index=edge_index)

    return loader, data_test


#for gene masking analyses
def custom_collate_fn_mask(batch):
    """
    Pytorch does not allow for geometric data so use this for making the masking collate fn

    Args:
    - batch (list of Data): A batch of Data objects to be collated.

    Returns:
    - Data: A new Data object containing x, edge index and masks
    """
    
    batch_data = [item for item in batch]
    x = torch.cat([data.x for data in batch_data], dim=1)
    edge_index = batch_data[0].edge_index
    #mask
    mask = torch.cat([data.mask for data in batch_data], dim=1)

    return Data(x=x, edge_index=edge_index, mask=mask)


def prepare_data_masking(scRNA, fraction_to_mask, test_size=0.2, batch_size = 32):
    """
    Function for the masking

    Args:
    - scRNA (pd.DataFrame): DataFrame containing single-cell RNA-seq data
    - fraction_to_mask (float): Fraction of genes to mask (set to 30%)
    - test_size (float): 0.2 portion of overall data

    Returns:
    - train_data (torch.Tensor): Training data tensor with a fraction of genes masked.
    - train_mask (torch.Tensor): Training mask tensor indicating which genes are masked (0) and unmasked (1).
    - test_data (torch.Tensor): Testing data tensor.
    - test_mask (torch.Tensor): Testing mask tensor.
    """

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(scRNA, test_size=test_size, random_state=42)

    # Convert training and testing data to tensors
    train_data_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_df.values, dtype=torch.float32)

    # Transpose the data tensors to have genes as rows and cells as columns
    train_data_tensor = train_data_tensor.T
    test_data_tensor = test_data_tensor.T

    # Determine the number of genes and the number of genes to mask
    num_genes = train_data_tensor.shape[1]
    num_genes_to_mask = int(num_genes * fraction_to_mask)

    # Randomly select genes to mask in the training data
    masked_genes_indices = np.random.choice(num_genes, num_genes_to_mask, replace=False)

    # Create mask tensor for the training data
    train_mask = torch.ones_like(train_data_tensor)
    train_mask[:, masked_genes_indices] = 0  # Set selected genes to zero in the mask

    # Apply the mask to the training data tensor
    dataset_train = [Data(x=train_data_tensor, edge_index=edge_index, mask=train_mask)]
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_mask)

    padded_test_df = pad_test_data(test_df, train_df)
    test_df =  padded_test_df.T
    test_df_tensor = torch.tensor(test_df.values, dtype = torch.float32)
    dummy_mask = torch.ones_like(test_df_tensor)
    test_dataset = [Data(x=test_df_tensor, edge_index=edge_index, mask=dummy_mask)]
    loader_test = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn_mask)

    return loader_train, loader_test

def test_model_full_reconstruction(model, test_loader, criterion):
    """
    Function for testing the masking data set on all scRNA_seq data

    Args:
    - model: The train model
    - test_loader: the testing data set with masked genes
    - criterion: mse
    Returns:
    - avg_test_loss: average mse VALUE
    """

    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No gradients needed for testing
        for batch in test_loader:
            output = model(batch)

            # Calculate loss on the full gene expression data
            loss = criterion(output, batch.x)
            total_loss += loss.item()
    
    avg_test_loss = total_loss / len(test_loader)
    return avg_test_loss


def test_model_on_masked_genes(model, test_loader, criterion, masked_genes_indices):
    """
    Function for testing the masking data set on only masked data that is unmasked in testing set

    Args:
    - model: The train model
    - test_loader: the testing data set with masked genes
    - criterion: mse
    Returns:
    - avg_test_loss: average mse VALUE
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No gradients needed for testing
        for batch in test_loader:
            output = model(batch)

            # Focus on the masked genes
            # Indexing rows instead of columns
            output_masked = output[masked_genes_indices, :]
            original_masked = batch.x[masked_genes_indices, :]

            # Calculate loss on masked genes
            loss = criterion(output_masked, original_masked)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)
    return avg_test_loss 


#for gene-peak autoencoder reconstruction
def prepare_data_for_extended_model(sc_joined, adj_matrix, fraction_to_mask, batch_size=32, test_size=0.2, random_state=42):
    """
    Prepare data and DataLoader for a GCN experiment with gene masking.

    Args:
    - sc_joined : DataFrame containing concatenated single-cell data scRNA and scATAC
    - adj_matrix : extended adj matrix
    - fraction_to_mask: 30% genes
    - batch_size (int): 32
    - test_size (float): 20% of overall set

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - test_data_tensor (torch.Tensor): Testing data tensor.
    """

    # Split the data into training and testing sets

    adjacency_matrix = torch.tensor(loaded_adj_matrix, dtype=torch.float32)
    edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
    
    train_df, test_df = train_test_split(sc_joined, test_size=0.2, random_state=42)
    
    padded_test_df = pad_test_data(test_df, train_df)
    
    scRNA_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    scRNA_tensor = scRNA_tensor.T
    num_genes = scRNA.shape[1]
    num_genes_to_mask = int(num_genes * 0.3)  # 20% of genes
    
    # Randomly select genes to mask
    np.random.seed(42)  # for reproducibility
    masked_genes_indices = np.random.choice(num_genes, num_genes_to_mask, replace=False)
    
    # Create a mask tensor
    mask = torch.ones_like(scRNA_tensor)
    
    mask[masked_genes_indices, :] = 0
    # Update the dataset
    # scRNA_tensor = scRNA_tensor.T
    # mask = mask.T

    dataset = [Data(x=scRNA_tensor, edge_index=edge_index, mask=mask)]
    batch_size = 32  # Define your batch size
    # DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_mask)

    #pad the test data and return it as a test loader

    test_df =  padded_test_df.T
    test_df_tensor = torch.tensor(test_df.values, dtype = torch.float32)
    dummy_mask = torch.ones_like(test_df_tensor)
    test_dataset = [Data(x=test_df_tensor, edge_index=edge_index, mask=dummy_mask)]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn_mask)
    
    return loader, test_loader



def train_model_extended_reconstruction(model, train_loader, epochs=800, learning_rate=0.001):
    """
    Train a extended model

    Args:
    model (torch.nn.Module): The extended model
    train_loader (DataLoader): training DatLoader
    epochs (int): Number of epochs
    learning_rate (float): Learning rate for the optimizer.

    Returns:
    list: List of epoch losses.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MSELoss()
    losses = []

    for epoch in tqdm(range(epochs)):  
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss_avg = epoch_loss / len(train_loader)
        losses.append(epoch_loss_avg)


    return losses

#for combined sigGCN model (base and extended)

def collate_fn_joint_model(batch):
   
    nn_data, gcn_data, labels = [], [], []
    for item in batch:
        nn_data_item, gcn_data_item, label_item = item
        nn_data.append(nn_data_item)
        # Assuming gcn_data_item is already a Data object; no need to append
        labels.append(torch.tensor(label_item, dtype=torch.long))

    nn_data = torch.stack(nn_data)
    labels = torch.stack(labels)

    # Assuming gcn_data is the same for all, so just use the first one from the batch
    gcn_data = batch[0][1]

    return nn_data, gcn_data, labels


def prepare_data_loader_combined(sc, labels, adj_matrix, test_size=0.2, random_state=42, batch_size=32, shuffle=True):
    """
    Prepare DataLoader for GCN training.

    Args:
    scRNA (DataFrame): Single-cell RNA-seq data.
    labels (Series or array-like): The labels for the data.
    adj_matrix (array-like): Adjacency matrix for the graph.
    test_size (float): Fraction of the dataset to be used as test data.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    batch_size (int): Batch size for the DataLoader.
    shuffle (bool): Whether to shuffle the data in the DataLoader.

    Returns:
    DataLoader: DataLoader for the training data.
    """

    # Split the data
    train_df, test_df, train_labels, test_labels = train_test_split(sc, labels, test_size=test_size, random_state=random_state)

    # Convert to PyTorch tensors
    nn_data = torch.tensor(train_df.values, dtype=torch.float32)
    gene_count_matrix = torch.tensor(train_df.T.values, dtype=torch.float32)
    adjacency_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()

    # Create GCN data
    gcn_data = Data(x=gene_count_matrix, edge_index=edge_index)

    # Create the combined dataset
    combined_dataset = CombinedDataset(nn_data, gcn_data, train_labels)

    # Prepare DataLoader
    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=True)

    nn_data_test = torch.tensor(test_df.values, dtype=torch.float32)

    # Prepare the testing GCN data
    # Assuming padding is needed as in the training phase
    padded_test_df = pad_test_data(test_df, train_df)
    gene_count_matrix = torch.tensor(padded_test_df.T.values, dtype = torch.float32)
    adjacency_matrix = torch.tensor(padded_test_df.T.values, dtype=torch.float32)
    edge_index = adjacency_matrix.nonzero(as_tuple = False).t().contiguous()
    gcn_data = Data(x =  gene_count_matrix, edge_index = edge_index)
    
    # If you have test labels, prepare them as well
    encoded_test_labels =test_labels
    
    # Create a DataLoader for testing
    combined_dataset = CombinedDataset(nn_data_test, gcn_data, encoded_test_labels)
    
    test_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, drop_last=True)

    return train_loader,test_loader
def train_combined_model(model, loader, epochs=75, learning_rate=0.001, weight_decay=0.01, device='cuda'):
    """
    Train a combined model (GCN and another NN) over multiple epochs.

    Args:
    model (torch.nn.Module): The combined model to be trained.
    loader (DataLoader): DataLoader containing the training data.
    epochs (int): Number of epochs to train the model.
    learning_rate (float): Learning rate for the optimizer.
    weight_decay (float): Weight decay (L2 penalty) for the optimizer.
    device (str): Device to run the training on ('cuda' or 'cpu').

    Returns:
    list: List of losses per epoch.
    list: Predicted outputs in the last epoch.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    total_accuracy = 0
    predicted_output = None

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        epoch_outputs = []
        for nn_data, gcn_data, labels in loader:
            nn_data, labels = nn_data.to(device), labels.to(device)
            gcn_data.x = gcn_data.x.to(device)
            gcn_data.edge_index = gcn_data.edge_index.to(device)

            optimizer.zero_grad()

            output, reconstructed_x = model(nn_data, gcn_data)

            classification_loss = F.nll_loss(output, labels)
            reconstruction_loss = F.mse_loss(reconstructed_x, gcn_data.x)
            loss = classification_loss + reconstruction_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted_labels = torch.max(output, 1)
            epoch_outputs.extend(predicted_labels.cpu().numpy())
            if epoch == epochs - 1:
                predicted_output = epoch_outputs

        losses.append(total_loss / len(loader))

    average_accuracy = total_accuracy / (len(loader) * epochs)
    return losses, predicted_output, average_accuracy

def test_combined_model(model, test_loader, device='cuda'):
    """
    Test a combined model (GCN and another NN) and compute loss and accuracy.

    Parameters:
    model (torch.nn.Module): The combined model to be tested.
    test_loader (DataLoader): DataLoader containing the test data.
    device (str): Device to run the testing on ('cuda' or 'cpu').

    Returns:
    tuple: Average classification loss, average reconstruction loss, and average accuracy.
    """
    model.eval()
    total_classification_loss = 0
    total_reconstruction_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for nn_data, gcn_data, labels in test_loader:
            nn_data, labels = nn_data.to(device), labels.to(device)
            gcn_data.x = gcn_data.x.to(device)
            gcn_data.edge_index = gcn_data.edge_index.to(device)

            # Forward pass
            output, reconstructed_x = model(nn_data, gcn_data)

            # Compute losses
            classification_loss = F.nll_loss(output, labels)
            reconstruction_loss = F.mse_loss(reconstructed_x[:17675, :], gcn_data.x[:17675, :])

            # Calculate accuracy
            accuracy = calculate_accuracy(output, labels)

            total_classification_loss += classification_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_accuracy += accuracy

    # Compute average losses and accuracy
    average_classification_loss = total_classification_loss / len(test_loader)
    average_reconstruction_loss = total_reconstruction_loss / len(test_loader)
    average_accuracy = total_accuracy / len(test_loader)

    return average_classification_loss, average_reconstruction_loss, average_accuracy




