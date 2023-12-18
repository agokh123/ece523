class NN(nn.Module):
    def __init__(self):
        #basic NN model used as a baseline
        super(NN, self).__init__()
        self.fc1 = nn.Linear( 17675, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 14)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1) 

def train_NN(net, optimizer, device, trainloader, criterion, num_epochs=5):
    #outline adaopted from ece523 homework written by Li Sun. 
    """
    Train a neural network

    Args:
    - net (nn.Module): The neural network to be trained.
    - optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
    - device (torch.device): The device to run training on (CPU or GPU).
    - trainloader (torch.utils.data.DataLoader): DataLoader containing the training data.
    - criterion (torch.nn.Module): The loss function used to compute the training loss.
    - num_epochs (int): Number of training epochs (default is 5).

    Returns:
    - net (nn.Module): The trained neural network.
    """

    #if cuda available move to GPU
    if torch.cuda.is_available():
        net.to(device)
    net.train()

    for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # Get the inputs and labels
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')
    return net

def test_NN(net, testloader):
    """
    Test a neural network on a test dataset and compute accuracy
    
    Args:
    - net (nn.Module): The trained neural network.
    - testloader (torch.utils.data.DataLoader): DataLoader containing the test data.

    Returns:
    - predicted_labels (list): List of predicted labels.
    - true_labels (list): List of true labels.
    """

    # Set the neural network to evaluation mode
    
    net.eval()
    total = 0
    correct = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #keep a list of the predicted labels so can compute confusion matrix later
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    print("Accuracy: ", 100 * correct // total)
    return predicted_labels, true_labels
    
def random_forest(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train and run a Random Forest classifier using scikit-learn

    Args:
    - X_train (pd.DataFrame): Training data features
    - y_train (array-like): Training data labels
    - X_test (pd.DataFrame): Testing data features
    - y_test (array-like): Testing data labels
    - random_state (int): Random seed for reproducibility (default is 42)

    Returns:
    - accuracy (float): Accuracy of the classifier on the test data
    """

    # Create a Random Forest classifier
    clf = RandomForestClassifier(random_state=random_state)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy on the test data
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy with random forest: ", accuracy)
    return accuracy




        
