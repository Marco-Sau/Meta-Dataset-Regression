"""
Author: Marco Sau
Date: February 2024

This script contains functions for visualizing data, particularly focusing on the distributions of numeric variables in datasets. 
It leverages libraries like numpy, seaborn, and matplotlib to create insightful and aesthetically pleasing visualizations. 

Functions:
1. plot_numeric_distributions
2. plot_relationships
3. plot_original_vs_interpolated
4. plot_bar_chart_model
5. plot_bar_chart_model_extra
6. plot_bar_chart_metric
7. plot_bar_chart_metric_extra
8. plot_line_chart
9. plot_line_chart_extra

Each function is designed to provide a clear and insightful visual representation of different types of data and metrics.
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from math import sqrt


def plot_numeric_distributions(dataframe):
    plots_per_row = 4
    """
    Plot the distribution of each numeric variable in the dataframe in a 4-column layout.

    :param dataframe: pandas DataFrame containing the dataset
    """
    # Select only numeric columns
    numeric_columns = dataframe.select_dtypes(include=['number']).columns

    # Number of rows, considering 4 histograms per row
    num_rows = len(numeric_columns) // plots_per_row + (1 if len(numeric_columns) % plots_per_row > 0 else 0)

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, num_rows * 4))
    axes = axes.ravel()  # Flatten the axes array

    for idx, column in enumerate(numeric_columns):
        sns.histplot(dataframe[column].dropna(), kde=False, bins=30, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {column}')
        axes[idx].set_xlabel(column)
        axes[idx].set_ylabel('Frequency')

    # Hide any empty subplots
    for i in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()



def plot_relationships(df, target, plots_per_row=4):
    """
    Plots the relationship between each numeric column in df (excluding the target) 
    and the target variable.

    :param df: DataFrame containing the data.
    :param target: The name of the target column.
    :param plots_per_row: Number of plots to display per row.
    """

    # Identify numeric columns (excluding the target column)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if target in numeric_columns: 
        numeric_columns.remove(target)

    # Calculate number of rows needed for 4 plots per row
    num_rows = len(numeric_columns) // plots_per_row + (1 if len(numeric_columns) % plots_per_row > 0 else 0)

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(20, 5 * num_rows))
    
    # If there is only one row, axes may not be an array
    if num_rows == 1:
        axes = [axes]

    axes = axes.ravel()  # Flatten the axes array

    for idx, col in enumerate(numeric_columns):
        sns.scatterplot(x=df[col], y=df[target], ax=axes[idx])
        axes[idx].set_title(f'Relationship between {col} and {target}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel(target)

    # Hide any empty subplots
    for i in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()



def plot_original_vs_interpolated(original_df, interpolated_df, column_name):
    """
    Plots the specified column from the original and interpolated DataFrames.

    Parameters:
    original_df (DataFrame): The original DataFrame.
    interpolated_df (DataFrame): The DataFrame after interpolation.
    column_name (str): The name of the column to be plotted.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(original_df[column_name], label='Original', color='blue')
    plt.plot(interpolated_df[column_name], label='Interpolated', color='red', linestyle='--')
    plt.title(f'Comparison of Original and Interpolated {column_name} Columns')
    plt.xlabel('Index')
    plt.ylabel(column_name)
    plt.legend()
    plt.show()
    
    
    
def plot_bar_chart_model(results_d1, results_d2, title_d1, title_d2):
    """
    Plots bar charts for two datasets to compare the performance of different models.

    This function creates two side-by-side bar charts. Each chart displays Mean Absolute Error (MAE),
    Mean Absolute Percentage Error (MAPE), and Symmetric Mean Absolute Percentage Error (SMAPE) for a
    series of models. The function is designed to allow for a visual comparison between two different datasets.

    Parameters:
    results_d1 (dict): A dictionary containing the performance metrics of models for the first dataset.
                       Expected keys are model names, and values are dictionaries with keys 'MAE', 'MAPE', and 'SMAPE'.
    results_d2 (dict): A dictionary containing the performance metrics of models for the second dataset.
                       Similar structure as results_d1.
    title_d1 (str): The title for the bar chart of the first dataset.
    title_d2 (str): The title for the bar chart of the second dataset.

    Each bar chart has models on the x-axis and the performance scores on the y-axis. The function automatically
    adjusts the position of the legends to avoid overlap with the bars. Each dataset's models and their respective
    scores are plotted as grouped bars.
    """
    models_d1 = list(results_d1.keys())
    mae_values_d1 = [results_d1[model]['MAE'] for model in models_d1]
    mape_values_d1 = [results_d1[model]['MAPE'] for model in models_d1]
    smape_values_d1 = [results_d1[model]['SMAPE'] for model in models_d1]

    models_d2 = list(results_d2.keys())
    mae_values_d2 = [results_d2[model]['MAE'] for model in models_d2]
    mape_values_d2 = [results_d2[model]['MAPE'] for model in models_d2]
    smape_values_d2 = [results_d2[model]['SMAPE'] for model in models_d2]

    x_d1 = np.arange(len(models_d1))
    x_d2 = np.arange(len(models_d2))
    width = 0.2  # width of the bars

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for D1
    ax1.bar(x_d1 - width, mae_values_d1, width, label='MAE')
    ax1.bar(x_d1, mape_values_d1, width, label='MAPE')
    ax1.bar(x_d1 + width, smape_values_d1, width, label='SMAPE')
    ax1.set_title(title_d1)
    ax1.set_xticks(x_d1)
    ax1.set_xticklabels(models_d1)
    ax1.set_ylabel('Scores')
    ax1.legend(bbox_to_anchor=(0.2, 1))


    # Plot for D2
    ax2.bar(x_d2 - width, mae_values_d2, width, label='MAE')
    ax2.bar(x_d2, mape_values_d2, width, label='MAPE')
    ax2.bar(x_d2 + width, smape_values_d2, width, label='SMAPE')
    ax2.set_title(title_d2)
    ax2.set_xticks(x_d2)
    ax2.set_xticklabels(models_d2)
    ax2.legend(bbox_to_anchor=(0.2, 1))

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()
    


def plot_bar_chart_model_extra(results_d1, results_d2, title_d1, title_d2):
    """
    Plots bar charts for two datasets to compare the performance of different models with extended metrics.

    This function creates two side-by-side bar charts for visual comparison of models' performance on two datasets.
    It displays Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute 
    Percentage Error (SMAPE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for each model.

    Parameters:
    results_d1 (dict): A dictionary with models as keys and their performance metrics as values for the first dataset.
                       Each model's dictionary should contain 'MAE', 'MAPE', 'SMAPE', 'MSE', and 'RMSE' values.
    results_d2 (dict): Similar to results_d1, but for the second dataset.
    title_d1 (str): Title for the bar chart of the first dataset.
    title_d2 (str): Title for the bar chart of the second dataset.

    The function plots a bar chart for each dataset with the models on the x-axis and the performance scores on the y-axis.
    It creates grouped bars for each model representing different metrics. The legends are placed appropriately to avoid
    overlapping with the bars.
    """
    models_d1 = list(results_d1.keys())
    mae_values_d1 = [results_d1[model]['MAE'] for model in models_d1]
    mape_values_d1 = [results_d1[model]['MAPE'] for model in models_d1]
    smape_values_d1 = [results_d1[model]['SMAPE'] for model in models_d1]
    mse_values_d1 = [results_d1[model]['MSE'] for model in models_d1]
    rmse_values_d1 = [sqrt(results_d1[model]['MSE']) for model in models_d1]  # Calculate RMSE

    models_d2 = list(results_d2.keys())
    mae_values_d2 = [results_d2[model]['MAE'] for model in models_d2]
    mape_values_d2 = [results_d2[model]['MAPE'] for model in models_d2]
    smape_values_d2 = [results_d2[model]['SMAPE'] for model in models_d2]
    mse_values_d2 = [results_d2[model]['MSE'] for model in models_d2]
    rmse_values_d2 = [sqrt(results_d2[model]['MSE']) for model in models_d2]  # Calculate RMSE

    x_d1 = np.arange(len(models_d1))  # positions for the bars
    x_d2 = np.arange(len(models_d2))
    width = 0.15  # width of the bars

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot for D1
    ax1.bar(x_d1 - 2*width, mae_values_d1, width, label='MAE')
    ax1.bar(x_d1 - width, mape_values_d1, width, label='MAPE')
    ax1.bar(x_d1, smape_values_d1, width, label='SMAPE')
    ax1.bar(x_d1 + width, mse_values_d1, width, label='MSE')
    ax1.bar(x_d1 + 2*width, rmse_values_d1, width, label='RMSE')
    ax1.set_title(title_d1)
    ax1.set_xticks(x_d1)
    ax1.set_xticklabels(models_d1)
    ax1.set_ylabel('Scores')
    ax1.legend()

    # Plot for D2
    ax2.bar(x_d2 - 2*width, mae_values_d2, width, label='MAE')
    ax2.bar(x_d2 - width, mape_values_d2, width, label='MAPE')
    ax2.bar(x_d2, smape_values_d2, width, label='SMAPE')
    ax2.bar(x_d2 + width, mse_values_d2, width, label='MSE')
    ax2.bar(x_d2 + 2*width, rmse_values_d2, width, label='RMSE')
    ax2.set_title(title_d2)
    ax2.set_xticks(x_d2)
    ax2.set_xticklabels(models_d2)
    ax2.legend()

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()

    
    
def plot_bar_chart_metric(results_d1, results_d2, title):
    """
    Plots a series of bar charts comparing two datasets across common metrics for each model.

    This function generates a subplot of bar charts, each representing a different model from the provided datasets.
    It compares the performance of these models based on Mean Absolute Error (MAE), Mean Absolute Percentage 
    Error (MAPE), and Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters:
    results_d1 (dict): A dictionary with models as keys and their performance metrics as values for the first dataset.
                       Each model's dictionary should contain 'MAE', 'MAPE', and 'SMAPE' values.
    results_d2 (dict): Similar to results_d1, but for the second dataset.
    title (str): The overall title for the series of subplots.

    Each subplot in the series represents a different model, with bars showing the performance metrics from both datasets.
    The function assumes that both datasets have the same models in the same order. Legends are included for clarity.
    """
    # Assuming both datasets have the same models in the same order
    models = list(results_d1.keys())
    metrics = ['MAE', 'MAPE', 'SMAPE']

    # Create subplots for each model
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, model in enumerate(models):
        # Extract metrics for this model from both datasets
        metrics_d1 = [results_d1[model][metric] for metric in metrics]
        metrics_d2 = [results_d2[model][metric] for metric in metrics]

        # X-axis positions
        x = np.arange(len(metrics))
        width = 0.35

        # Plot for each model
        axes[i].bar(x - width/2, metrics_d1, width, label='D1')
        axes[i].bar(x + width/2, metrics_d2, width, label='D2')

        axes[i].set_title(model)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(metrics)
        axes[i].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    
    
def plot_bar_chart_metric_extra(results_d1, results_d2, title):
    """
    Plots a series of bar charts for each model, comparing additional metrics across two datasets.

    This function generates a subplot of bar charts where each chart corresponds to a different model from the 
    provided datasets. It compares the performance of these models based on an extended set of metrics: Mean Absolute 
    Error (MAE), Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute Percentage Error (SMAPE), Mean Squared 
    Error (MSE), and Root Mean Squared Error (RMSE).

    Parameters:
    results_d1 (dict): A dictionary with models as keys and their performance metrics as values for the first dataset.
                       Each model's dictionary should contain 'MAE', 'MAPE', 'SMAPE', 'MSE', and 'RMSE' values.
    results_d2 (dict): Similar to results_d1, but for the second dataset.
    title (str): The overall title for the series of subplots.

    The function creates a subplot for each model. In each subplot, two sets of bars represent the performance 
    metrics from both datasets. It is assumed that both datasets contain the same models in the same order. 
    Legends are provided for clear distinction between datasets.
    """
    # Assuming both datasets have the same models in the same order
    models = list(results_d1.keys())
    metrics = ['MAE', 'MAPE', 'SMAPE', 'MSE', 'RMSE']

    # Create subplots for each model
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, model in enumerate(models):
        # Extract metrics for this model from both datasets
        metrics_d1 = [results_d1[model][metric] for metric in metrics]
        metrics_d2 = [results_d2[model][metric] for metric in metrics]

        # X-axis positions
        x = np.arange(len(metrics))
        width = 0.35

        # Plot for each model
        axes[i].bar(x - width/2, metrics_d1, width, label='D1')
        axes[i].bar(x + width/2, metrics_d2, width, label='D2')

        axes[i].set_title(model)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(metrics)
        axes[i].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    


def plot_line_chart(results_d1, results_d2, title_d1, title_d2):
    """
    Plots the performance metrics (MAE, MAPE, SMAPE) of various models for two separate datasets.

    This function creates line plots for two datasets, visually comparing the performance of different models 
    based on three metrics: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and 
    Symmetric Mean Absolute Percentage Error (SMAPE). Each dataset is represented in a separate subplot.

    Parameters:
    results_d1 (dict): A dictionary containing the performance metrics of models for the first dataset. 
                       The keys are model names, and the values are dictionaries with keys 'MAE', 'MAPE', and 'SMAPE'.
    results_d2 (dict): Similar to results_d1, but containing metrics for the second dataset.
    title_d1 (str): Title for the plot corresponding to the first dataset.
    title_d2 (str): Title for the plot corresponding to the second dataset.

    Each subplot displays line plots with different markers and line styles for each metric. The x-axis 
    represents the models, and the y-axis represents the score of the metrics. The function assumes that 
    the order and number of models are the same in both datasets.

    Returns:
    None: The function plots the graphs but does not return any value.
    """
    models_d1 = list(results_d1.keys())
    mae_values_d1 = [results_d1[model]['MAE'] for model in models_d1]
    mape_values_d1 = [results_d1[model]['MAPE'] for model in models_d1]
    smape_values_d1 = [results_d1[model]['SMAPE'] for model in models_d1]

    models_d2 = list(results_d2.keys())
    mae_values_d2 = [results_d2[model]['MAE'] for model in models_d2]
    mape_values_d2 = [results_d2[model]['MAPE'] for model in models_d2]
    smape_values_d2 = [results_d2[model]['SMAPE'] for model in models_d2]

    x_d1 = np.arange(len(models_d1))
    x_d2 = np.arange(len(models_d2))

    # Create the line plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for D1
    ax1.plot(x_d1, mae_values_d1, marker='o', label='MAE', linestyle='-')
    ax1.plot(x_d1, mape_values_d1, marker='s', label='MAPE', linestyle='--')
    ax1.plot(x_d1, smape_values_d1, marker='^', label='SMAPE', linestyle='-.')
    ax1.set_title(title_d1)
    ax1.set_xticks(x_d1)
    ax1.set_xticklabels(models_d1, rotation=45)
    ax1.set_ylabel('Scores')
    ax1.legend()

    # Plot for D2
    ax2.plot(x_d2, mae_values_d2, marker='o', label='MAE', linestyle='-')
    ax2.plot(x_d2, mape_values_d2, marker='s', label='MAPE', linestyle='--')
    ax2.plot(x_d2, smape_values_d2, marker='^', label='SMAPE', linestyle='-.')
    ax2.set_title(title_d2)
    ax2.set_xticks(x_d2)
    ax2.set_xticklabels(models_d2, rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    


def plot_line_chart_extra(results_d1, results_d2, title_d1, title_d2):
    """
    Plots line charts for two datasets, comparing extended performance metrics (MAE, MAPE, SMAPE, MSE, RMSE) for various models.

    This function creates two line plots, one for each dataset, to visually compare the performance of different models 
    based on an expanded set of metrics: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Symmetric Mean 
    Absolute Percentage Error (SMAPE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Each dataset is 
    represented in a separate subplot.

    Parameters:
    results_d1 (dict): A dictionary containing the performance metrics of models for the first dataset. 
                       Keys are model names, and values are dictionaries with metrics as keys ('MAE', 'MAPE', 'SMAPE', 'MSE', 'RMSE').
    results_d2 (dict): Similar to results_d1, but for the second dataset.
    title_d1 (str): Title for the plot corresponding to the first dataset.
    title_d2 (str): Title for the plot corresponding to the second dataset.

    Each subplot shows line plots for each metric, with different markers for each metric. The x-axis represents the models, 
    and the y-axis shows the metric scores. The function assumes that the order and number of models are the same in both datasets.
    """
    models_d1 = list(results_d1.keys())
    mae_values_d1 = [results_d1[model]['MAE'] for model in models_d1]
    mape_values_d1 = [results_d1[model]['MAPE'] for model in models_d1]
    smape_values_d1 = [results_d1[model]['SMAPE'] for model in models_d1]
    mse_values_d1 = [results_d1[model]['MSE'] for model in models_d1]
    rmse_values_d1 = [sqrt(results_d1[model]['MSE']) for model in models_d1]  # Calculate RMSE

    models_d2 = list(results_d2.keys())
    mae_values_d2 = [results_d2[model]['MAE'] for model in models_d2]
    mape_values_d2 = [results_d2[model]['MAPE'] for model in models_d2]
    smape_values_d2 = [results_d2[model]['SMAPE'] for model in models_d2]
    mse_values_d2 = [results_d2[model]['MSE'] for model in models_d2]
    rmse_values_d2 = [sqrt(results_d2[model]['MSE']) for model in models_d2]  # Calculate RMSE

    x_d1 = np.arange(len(models_d1))
    x_d2 = np.arange(len(models_d2))

    # Create the line plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for D1
    ax1.plot(x_d1, mae_values_d1, marker='o', label='MAE')
    ax1.plot(x_d1, mape_values_d1, marker='s', label='MAPE')
    ax1.plot(x_d1, smape_values_d1, marker='^', label='SMAPE')
    ax1.plot(x_d1, mse_values_d1, marker='x', label='MSE')
    ax1.plot(x_d1, rmse_values_d1, marker='*', label='RMSE')
    ax1.set_title(title_d1)
    ax1.set_xticks(x_d1)
    ax1.set_xticklabels(models_d1, rotation=45)
    ax1.set_ylabel('Scores')
    ax1.legend()

    # Plot for D2
    ax2.plot(x_d2, mae_values_d2, marker='o', label='MAE')
    ax2.plot(x_d2, mape_values_d2, marker='s', label='MAPE')
    ax2.plot(x_d2, smape_values_d2, marker='^', label='SMAPE')
    ax2.plot(x_d2, mse_values_d2, marker='x', label='MSE')
    ax2.plot(x_d2, rmse_values_d2, marker='*', label='RMSE')
    ax2.set_title(title_d2)
    ax2.set_xticks(x_d2)
    ax2.set_xticklabels(models_d2, rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.show()