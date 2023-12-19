def get_df(paths):
    """
    Create a pandas DataFrame of either modality (scRNA-seq/ scATAC-seq) from the provided Matrix Market type from R.

    Args:
    - paths (list of str): A list of file paths.
        - paths[0] (str): File path to the matrix in Matrix Market format.
        - paths[1] (str): File path to the row names file.
        - paths[2] (str): File path to the column names file, if available.

    Returns:
    - pd.DataFrame: A DataFrame containing the matrix data with row and column labels.
    """

    # Extract file paths and conditionally check if column paths are provided
    mat_file = paths[0]
    row_file = paths[1]
    if len(paths) > 2:
        col_file = paths[2]
    
    # Read the sparse matrix from the Matrix Market format and variable names and convert to NumPy arrays
    sparsematrix = io.mmread(mat_file)
    m_dense = sparsematrix.toarray()
    var_names = np.genfromtxt(row_file, dtype=str)
    
    # Check if a column names file is provided and create/return the dataframe with cells as rows and peaks/genes as columns
    if len(paths) > 2:
        col_names = np.genfromtxt(col_file, dtype=str)
        
        # Create a DataFrame with column names
        sc_df = pd.DataFrame(m_dense, columns=col_names, index=var_names)
    else:
        # Create a DataFrame without column names
        sc_df = pd.DataFrame(m_dense, index=var_names)
    
    return sc_df

def process_scRNA_data(scRNA):
    """
    Convert single-cell RNA-seq gene names (columns) from Ensembl IDs to gene symbols.

    Args:
    - scRNA (pd.DataFrame): A DataFrame containing single-cell RNA-seq data.

    Returns:
    - pd.DataFrame: Processed scRNA-seq data with gene symbols as column names.
    """

    #initliaze the mygene package 
    mg = mygene.MyGeneInfo()

    #get mappings for each column (gene) in the scRNA_seq matrix 
    gene_info = mg.querymany(scRNA.columns, scopes="ensembl.gene", fields="symbol", as_dataframe=True)
    gene_names_dict = gene_info["symbol"].fillna(gene_info["_id"]).to_dict()
    gene_names = [gene_names_dict.get(ensembl_id, ensembl_id) for ensembl_id in scRNA.columns]
    scRNA.columns = (str(gene_) for gene_ in gene_names)

    #some genes dont get converted because they dont have mappings in the mygene database so drop them
    columns_to_drop = [col for col in scRNA.columns if col.startswith("ENSG0")]
    scRNA = scRNA.drop(columns=columns_to_drop)
    return scRNA

def get_data_NN(scRNA, scATAC, labels):
    """
    Prepare data for a neural network

    Args:
    - scRNA (pd.DataFrame): DataFrame containing single-cell RNA-seq data
    - scATAC (pd.DataFrame): DataFrame containing single-cell ATAC-seq data
    - labels (list or array-like): Labels or target values for the data

    Returns:
    - trainloader (torch.utils.data.DataLoader): Training DataLoader
    - valloader (torch.utils.data.DataLoader): Validation DataLoader
    - testloader (torch.utils.data.DataLoader): Testing DataLoader
    """
    #one hot encode labels
    label_encoder = LabelEncoder()
    encoded_types = label_encoder.fit_transform(labels)

    #make both data types into a tensor
    sc_tensor = torch.tensor(data.to_numpy(), dtype=torch.float32)
    encoded_type_tensor = torch.tensor(encoded_types, dtype=torch.int64)
    full_dataset = torch.utils.data.TensorDataset(sc_tensor, encoded_type_tensor)


    #split into train, test and validation the paper did a 80-10-10 split
    train_size = int(0.8 * len(full_dataset))
    val_test_size = (len(full_dataset) - train_size)/2

    #split the data with a 80/10/10 split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, int(val_test_size-.5), int(val_test_size+.5)])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers = 2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, num_workers = 2)
    testloader =  torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers = 2)
    
    return trainloader, valloader, testloader
        
        
        
def maslov_sneppen_rewire_one_swap_per_iter(adj_matrix, num_iterations= 100000000):
    """
    Rewire edges in a network using the Maslov-Sneppen model with one swap per iteration.

    Args:
    - adj_matrix (numpy.ndarray): The adjacency matrix of the network.
    - num_iterations (int, optional): The number of iterations to perform (default is set to 100000000).

    Returns:
    - numpy.ndarray: The Maslov-Sneppen rewired adjacency matrix.
    """
    
    num_nodes = adj_matrix.shape[0]
    edges = np.argwhere(adj_matrix)  # Get all edges as an array of pairs

    for _ in tqdm(range(num_iterations)):
        # Randomly select two edges (a-b and c-d)
        idx1, idx2 = random.sample(range(len(edges)), 2)
        a, b = edges[idx1]
        c, d = edges[idx2]

        # Check for self-loops and duplicate edges
        if a != c and b != d and not adj_matrix[a, d] and not adj_matrix[c, b]:
            # Rewire the edges
            adj_matrix[a, b] = adj_matrix[b, a] = 0
            adj_matrix[c, d] = adj_matrix[d, c] = 0
            adj_matrix[a, d] = adj_matrix[d, a] = 1
            adj_matrix[c, b] = adj_matrix[b, c] = 1

            # Update the edges list
            edges[idx1] = [a, d]
            edges[idx2] = [c, b]

    return adj_matrix

def jaccard_similarity_matrix_pytorch(df, chunk_size=100):
    """
    Compute a Jaccard similarity matrix for peak-peak edges this is GPU supported as it takes a long time

    df: scATAC_df
    param chunk_size: Number of rows to process in each chunk this is gpu supported
    Return: A DataFrame representing the Jaccard similarity matrix.
    """
    n_peaks = df.shape[0]
    similarity_matrix = np.zeros((n_peaks, n_peaks))

    # Ensure the DataFrame is binary and convert to a PyTorch tensor
    data_tensor = torch.tensor(df.values.astype(int)).float()

    if torch.cuda.is_available():
        data_tensor = data_tensor.cuda()

    for start_row in tqdm(range(0, n_peaks, chunk_size)):
        end_row = min(start_row + chunk_size, n_peaks)

        # Processing a chunk
        chunk = data_tensor[start_row:end_row]

        for i in range(n_peaks):
            peak = data_tensor[i]

            intersection = torch.sum((chunk * peak) > 0, dim=1).float()
            union = torch.sum((chunk + peak) > 0, dim=1).float()

            # Compute Jaccard similarity for the chunk
            chunk_similarity = intersection / union

            # Store the result
            similarity_matrix[start_row:end_row, i] = chunk_similarity.cpu().numpy()

    return pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

def generate_adjacency_matrix(gene_names, known_interactions):
    """
    Generate an adjacency matrix for given gene interactions.

    Args:
    - gene_names (list): A list of gene names.
    - known_interactions (pd.DataFrame): A DataFrame containing known gene interactions, with columns 'Gene_A' and 'Gene_B'.

    Returns:
    - numpy.ndarray: An adjacency matrix representing the interactions.
    """
    adj_matrix = np.zeros((len(gene_names), len(gene_names)))
    for _, row in tqdm(known_interactions.iterrows()):
        i, j = gene_names.index(row['Gene_A']), gene_names.index(row['Gene_B'])
        adj_matrix[i, j] = 1
    
    return adj_matrix

def preprocess_joined_data(scRNA, scATAC, loaded_adj_matrix, num_peaks_to_keep=15000):
    """
    Function to concatenate adjancecny matrix and make the new enhanced gene network

    Args:
    - scRNA : scRNA dataframe
    - scATAC : scATAC dataframe
    - loaded_adj_matrix: enhanced adjancey matix
    - num_genes_to_keep: 15000 genes kept

    Returns:
    - sc_joined : joined scATAC/scRNA frame
    - trimmed_adj_matrix (numpy.ndarray): Corresponding enhanced gene matrix
    """

    # Concatenate scRNA and scATAC DataFrames
    sc_joined = pd.concat([scRNA, scATAC], axis=1)
    num_columns_to_keep = len(sc_joined.columns) - num_genes_to_keep
    sc_joined = sc_joined.iloc[:, :num_columns_to_keep]

    # Trim the overall network
    num_rows_to_keep = loaded_adj_matrix.shape[0] - num_genes_to_keep
    num_columns_to_keep = loaded_adj_matrix.shape[1] - num_genes_to_keep
    trimmed_adj_matrix = loaded_adj_matrix[:num_rows_to_keep, :num_columns_to_keep]

    return sc_joined, trimmed_adj_matrix

