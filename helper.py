# In this file we create our helper function for plotting the training proccess

import matplotlib.pyplot as plt
from IPython import display # IPython is what we will be using for our display


plt.ion() # Enables interactive mode on matplotlib, this allows our scores display to be constantly updated


def plot(scores, mean_scores):
    display.clear_output(wait = True) # Clears the new output only when the new output is ready to be displayed
    display.display(plt.gcf()) # Gets the latest figure and sets as display
    plt.clf() # Clears current figure after display
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin = 0) # Set y limit of current axis
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block = False)
    plt.pause(.1)