import numpy as np
from statistics import mode
import sys

# Find the labels of the k-nearest neighbors of a given data point.
def find_near_labels(X, Dataset, Labels, k):
    """
    Parameters:
    - X: The data point whose neighbors are to be found.
    - Dataset: The dataset containing all data points.
    - Labels: The labels corresponding to the data points in the dataset.
    - k: The number of nearest neighbors to consider.
    Returns:
    - nearest_labels: The labels of the k-nearest neighbors.
    """
    
    # Calculating the distances from a particular point X to each point of a dataset
    Difference = X - Dataset
    Distances = np.sum(Difference**2, axis=1)
    # Getting the index of the k-nearest neighbors
    sorted_ids = np.argsort(Distances)
    nearest_labels = Labels[sorted_ids[0:k]]
    
    return nearest_labels

# Predict the labels of the test dataset based on the k-nearest neighbors algorithm
def knn(Test_Dataset, Train_Dataset, Train_Labels, k):
    """
    Parameters:
    - Test_Dataset: The dataset to be classified.
    - Train_Dataset: The dataset used for classification.
    - Train_Labels: The labels corresponding to the training dataset.
    - k: The number of nearest neighbors to consider for classification.
    Returns:
    - Predicted_Labels: The predicted labels for the test dataset.
    """
    
    Predicted_Labels = []
    # Iterate over each data point and find the nearest neighbors
    for i in range(len(Test_Dataset)):
        # Find the classes of the k nearest neighbors for each point
        Labels = find_near_labels(Test_Dataset[i], Train_Dataset, Train_Labels, k) 
        try:
            # Predicting the label based on the majority class of the k-nearest neighbors
            Predicted_Labels.append(mode(Labels)) 
        except ValueError:
            sys.exit("Please enter a different number of neighbors")
            
    return np.array(Predicted_Labels)
