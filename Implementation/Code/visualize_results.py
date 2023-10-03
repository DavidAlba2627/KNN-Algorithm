import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import numpy as np

# Visualize the predictions made by the KNN algorithm.
def plot_predictions(X, y_pred, title="Predictions"):
    """
    Parameters:
    - X: The dataset.
    - y_pred: The predicted labels.
    - title: The title of the plot.
    """
    color_map = ListedColormap(['mediumseagreen', 'mediumblue'])
    
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=color_map, edgecolor='k', s=20)
    plt.title(title, fontsize=18)

    plt.show()

# Compare the predicted labels with the correct labels
def compare_with_correct(X_test, y_test, y_pred):
    """
    Parameters:
    - X_test: The test dataset.
    - y_test: The correct labels.
    - y_pred: The predicted labels.
    """
    
    color_map = ListedColormap(['mediumseagreen', 'mediumblue'])
    # Creating a figure and a set of subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # Plotting the test set with the correct labels
    scatter = ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=color_map, alpha=0.4)
    ax[0].legend(handles=scatter.legend_elements()[0], labels=['0', '1'], title='Classes')
    ax[0].set_title("Correct Classes", fontsize=18)
    # Creating a new color map for the predicted labels
    color_map2 = ListedColormap(['mediumseagreen', 'mediumblue', 'firebrick'])
    # Finding the indices of the points that were classified incorrectly
    incorrect_indices = np.where(y_pred != y_test)[0]
    # The labels of the points that were classified incorrectly are set to 2
    y_plot = y_pred.copy()
    y_plot[incorrect_indices] = 2
    # Plotting the test set with the predicted labels
    scatter = ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_plot, s=50, cmap=color_map2, alpha=0.4)
    ax[1].legend(handles=scatter.legend_elements()[0], labels=['0', '1', 'Incorrect'], title='Classes')
    ax[1].set_title("Predicted Classes", fontsize=18)
    
    plt.show()

# Visualize the confusion matrix using a heatmap
def visualize_confusion_matrix(y_true, y_pred):
    """
    Parameters:
    - y_true: The true labels.
    - y_pred: The predicted labels.
    """
    
    # Creating a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Creating a figure
    plt.figure(figsize=(8, 5))
    # Visualizing the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=['0', '1'],
                yticklabels=['0', '1'],
                cmap="YlOrBr", 
                annot_kws={"size": 18})
    plt.title('Confusion Matrix', fontsize=18)
    ax.set_xlabel('Predicted labels', fontsize=16)
    ax.set_ylabel('True labels', fontsize=16)
    ax.set_xticklabels(['Class 0', 'Class 1'], fontsize=14)
    ax.set_yticklabels(['Class 0', 'Class 1'], fontsize=14)
  
    plt.show()
