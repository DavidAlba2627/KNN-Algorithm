import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create a synthetic dataset, split it into training and testing sets, and standardize it
def generate_dataset(n_samples=3500, noise=0.2, test_size=0.25, random_state=0):
    """  
    Parameters:
    - n_samples: Total number of samples in the dataset.
    - noise: Standard deviation of Gaussian noise added to the data.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Random state seed for reproducibility. 
    Returns:
    - X_train, X_test: Training and testing data.
    - y_train, y_test: Labels for training and testing data.
    """
    
    # Creating a synthetic dataset
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Standardizing the train dataset
    sc_train = StandardScaler()
    X_train = sc_train.fit_transform(X_train)
    # Standardizing the test dataset
    sc_test= StandardScaler()
    X_test = sc_test.fit_transform(X_test)
    
    return X_train, X_test, y_train, y_test
