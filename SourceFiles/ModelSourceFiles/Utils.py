import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PIL
from IPython import display

## These two functions take model history and display graphs for loss and graphs for accuracy, they are saved to disk
## I use seaborn as a plotting API because its my fave
def Show_and_save_loss_graph(model_history, name, title):
    epochs = np.arange(1, len(model_history.history['Discriminator_Generator_Loss']) + 1)#Epoch index
    data = {"Discriminator Loss Generated": model_history.history['Discriminator_Generator_Loss'],
            "Discriminator Loss Real": model_history.history['Discriminator_Real_Loss'],
            "Generator Loss": model_history.history['Generator_Loss'], 'Epoch': epochs} ##create dictionary of data for pandas
    data = pd.DataFrame(data) #create dataframe
    fig, ax = plt.subplots(figsize=[12, 6])
    plot = sns.lineplot(data=pd.melt(data, ['Epoch']), y='value', x='Epoch', hue='variable')#create lineplot of metrics by melting to Epochs
    plot.set_title(title)
    plot.set_ylabel("Loss")
    plot.set_xlabel("Epoch")
    plot.legend_.set_title('Loss Source')
    plt.show() #show for notebooks
    fig_name = name + '_loss_graph'
    fig.savefig(fig_name, dpi=600, bbox_inches="tight") #save

## This function is identical to above
def Show_and_save_accuracy_graph(model_history, name, title):
    epochs = np.arange(1, len(model_history.history['Discriminator_Generator_Accuracy']) + 1)
    data = {"Discriminator Accuracy Generated": model_history.history['Discriminator_Generator_Accuracy'],
            "Discriminator Accuracy Real": model_history.history['Discriminator_Real_Accuracy'],
             'Epoch': epochs}
    data = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=[12, 6])
    plot = sns.lineplot(data=pd.melt(data, ['Epoch']), y='value', x='Epoch', hue='variable')
    plot.set_title(title)
    plot.set_ylabel("Accuracy")
    plot.set_xlabel("Epoch")
    plot.legend_.set_title('Image Type Accuracy')
    plt.show()
    fig_name = name + '_accuracy_graph'
    fig.savefig(fig_name, dpi=600, bbox_inches="tight")