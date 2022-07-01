''' 
Author: Minyoung Kim ( Jul. 1, 2022)
2022 Hackaton 
Team: ThankQ
description: 
    plot functions for matrix 
    This is used to plot adjacency matrix
''' 


import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot(cm:np.array, labels= True, normalize = False):
    """plot 2d matrix using colormap blues 

    Args:
        cm (np.array): matrix to plot 
        labels (bool, optional): label the data with number. Defaults to True.
        normalize (bool, optional): whether normalize the value . Defaults to False.
    """
    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")