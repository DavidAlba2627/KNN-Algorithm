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
    # Creating a figure
    fig, ax = plt.subplots(figsize=(8,5))
    # Creating a scatter plot
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap=color_map, alpha = 0.4)
    # Adding a legend to the plot
    legend = ax.legend(handles=scatter.legend_elements()[0], labels=['0', '1'], title = 'Classes', fontsize = 11)
    legend.get_title().set_fontsize(13)
    ax.set_title(title, fontsize=18)    

    plt.show()

# Compare the predicted labels with the correct labels
def compare_with_correct(X, y_true, y_pred):
    """
    Parameters:
    - X: The dataset.
    - y_true: The correct labels.
    - y_pred: The predicted labels.
    """
    
    color_map = ListedColormap(['mediumseagreen', 'mediumblue'])
    labels = ['0', '1']
    y_plot = y_pred.copy()
    # Creating a figure and a set of subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # Plotting the test set with the correct labels
    scatter = ax[0].scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap=color_map, alpha=0.4)
    legend = ax[0].legend(handles=scatter.legend_elements()[0], labels = labels, title='Classes', fontsize = 11)
    legend.get_title().set_fontsize(13)
    ax[0].set_title("Correct Classes", fontsize=18)
    # Finding the indices of the points that were classified incorrectly
    incorrect_indices = np.where(y_pred != y_true)[0]
    
    # Check if there are misclassified points
    if incorrect_indices.sum() > 0:
        # Creating a new color map for the predicted labels
        color_map = ListedColormap(['mediumseagreen', 'mediumblue', 'firebrick'])
        # The labels of the points that were classified incorrectly are set to 2
        y_plot[incorrect_indices] = 2
        labels = ['0', '1', 'Misclassified']
        
    # Plotting the test set with the predicted labels
    scatter = ax[1].scatter(X[:, 0], X[:, 1], c=y_plot, s=50, cmap=color_map, alpha=0.4)
    legend = ax[1].legend(handles=scatter.legend_elements()[0], labels = labels, title='Classes', fontsize = 11)
    legend.get_title().set_fontsize(13)
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
    plt.xlabel('Predicted labels', fontsize=16)
    plt.ylabel('True labels', fontsize=16)
    plt.xticks([0.5, 1.5], ['Class 0', 'Class 1'], fontsize=14)
    plt.yticks([0.5, 1.5], ['Class 0', 'Class 1'], fontsize=14)
  
    plt.show()
