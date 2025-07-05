import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def categorical_distribution(dataframe, column_name, title=None, save=False, save_path=None, save_format='png', save_dpi=100):
    """
    Plots the distribution of a categorical column with counts on top of bars.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the categorical column to plot.
        title (str, optional): The title of the plot. Defaults to None.
    """
    plt.figure(figsize=(15, 6), dpi=save_dpi) # Adjusted figure size for better readability
    # if title: plt.suptitle(title, fontsize=20) # Used suptitle for overall title

    # Create the countplot
    ax = sns.countplot(y=column_name, data=dataframe, order=dataframe[column_name].value_counts().index, palette='viridis', stat="percent")

    # Add count labels on top of each bar
    for p in ax.patches:
        width = p.get_width()    # Get the width of the bar (which is the count)
        plt.text(width + 0.5,     # Set the x position of the text slightly to the right of the bar
                 p.get_y() + p.get_height() / 2, # Set the y position at the middle of the bar
                 '{:0.2f}%'.format(width), # Format the count to an integer
                 ha='left', va='center', fontsize=10, color='black') # Text alignment and style

    plt.xlabel('Percent', fontsize=12) # Label for x-axis
    plt.ylabel(column_name.replace('_', ' ').title(), fontsize=12) # Formatted label for y-axis
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap

    if save: plt.savefig(f"../../datasets/Skoltech_Anomaly_Benchmark/plots/{save_path}/{title}.{save_format}", format=save_format)
    plt.show()