import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from create_dataset import generate_dataset
from knn_function import knn as KNeighborsClass
from visualize_results import plot_predictions, compare_with_correct, visualize_confusion_matrix

def main():
    # Creating and standardizing the dataset
    X_train, X_test, y_train, y_test = generate_dataset(n_samples=3500, noise=0.2, test_size=0.25, random_state=0)
  
    # Applying the KNN algorithm
    y_pred = KNeighborsClass(X_test, X_train, y_train, k=5)

    # Visualizing the predictions
    plot_predictions(X_test, y_pred, title="Predictions with KNN")

    # Comparing predictions with correct labels
    compare_with_correct(X_test, y_test, y_pred)

    # Visualizing the confusion matrix
    visualize_confusion_matrix(y_test, y_pred)
  
    # Printing the accuracy
    print("The accuracy is:", accuracy_score(y_test, y_pred))
  
if __name__ == "__main__":
    main()
