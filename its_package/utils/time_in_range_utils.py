#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time In Range Utilities
----------------------
Utilities for calculating and analyzing glucose time in range (TIR) metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def calculate_time_in_range(glucose_data, low_threshold=80, high_threshold=130):
    """
    Calculate time in range metrics for glucose data.
    
    Parameters:
    -----------
    glucose_data : pandas.Series or numpy.ndarray
        Glucose values
    low_threshold : float
        Lower bound of the target range (default: 80 mg/dL)
    high_threshold : float
        Upper bound of the target range (default: 130 mg/dL)
        
    Returns:
    --------
    dict
        Dictionary containing time in range metrics
    """
    # Convert to numpy array if it's a pandas Series
    if isinstance(glucose_data, pd.Series):
        values = glucose_data.values
    else:
        values = np.asarray(glucose_data)
    
    # Calculate metrics
    in_range = np.logical_and(values >= low_threshold, values <= high_threshold)
    below_range = values < low_threshold
    above_range = values > high_threshold
    
    # Calculate percentages
    total_count = len(values)
    if total_count > 0:
        tir_percent = 100 * np.sum(in_range) / total_count
        below_percent = 100 * np.sum(below_range) / total_count
        above_percent = 100 * np.sum(above_range) / total_count
    else:
        tir_percent = 0.0
        below_percent = 0.0
        above_percent = 0.0
    
    # Calculate mean glucose and other stats
    mean_glucose = np.mean(values)
    std_glucose = np.std(values)
    min_glucose = np.min(values)
    max_glucose = np.max(values)
    
    # Return all metrics
    return {
        'tir': tir_percent,
        'below_range': below_percent,
        'above_range': above_percent,
        'mean_glucose': mean_glucose,
        'std_glucose': std_glucose,
        'min_glucose': min_glucose,
        'max_glucose': max_glucose
    }

def plot_time_in_range(glucose_data, time_index=None, low_threshold=80, high_threshold=130, 
                       intervention_time=None, title=None, figsize=(10, 6), ax=None):
    """
    Plot glucose data with time in range highlighted.
    
    Parameters:
    -----------
    glucose_data : pandas.Series
        Glucose values
    time_index : pandas.DatetimeIndex, optional
        Time index for the glucose data (if not included in Series)
    low_threshold : float
        Lower bound of the target range (default: 80 mg/dL)
    high_threshold : float
        Upper bound of the target range (default: 130 mg/dL)
    intervention_time : pandas.Timestamp, optional
        Time of intervention to mark on plot
    title : str, optional
        Title for the plot
    figsize : tuple
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the plot
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get x-axis values
    if time_index is not None:
        x = time_index
    elif isinstance(glucose_data.index, pd.DatetimeIndex):
        x = glucose_data.index
    else:
        x = np.arange(len(glucose_data))
    
    # Plot the data
    ax.plot(x, glucose_data, 'b-', linewidth=2, label='Glucose')
    
    # Highlight the time in range area
    ax.axhspan(low_threshold, high_threshold, alpha=0.2, color='green', label=f'Target Range ({low_threshold}-{high_threshold} mg/dL)')
    
    # Mark intervention time if provided
    if intervention_time is not None:
        ax.axvline(x=intervention_time, color='r', linestyle='--', linewidth=2, label='Intervention')
    
    # Add threshold lines
    ax.axhline(y=low_threshold, color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=high_threshold, color='orange', linestyle='--', alpha=0.7)
    
    # Calculate TIR metrics
    metrics = calculate_time_in_range(glucose_data, low_threshold, high_threshold)
    
    # Add metrics text box
    metrics_text = f"Time in Range: {metrics['tir']:.1f}%\n" \
                  f"Below Range: {metrics['below_range']:.1f}%\n" \
                  f"Above Range: {metrics['above_range']:.1f}%\n" \
                  f"Mean Glucose: {metrics['mean_glucose']:.1f} mg/dL"
    
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Glucose (mg/dL)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Glucose Time in Range Analysis')
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Set reasonable y-limits with padding
    y_min = max(0, metrics['min_glucose'] - 20)
    y_max = metrics['max_glucose'] + 20
    ax.set_ylim(y_min, y_max)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_tir_by_dose(doses, tir_values, low_threshold=80, high_threshold=130, 
                     optimal_dose=None, figsize=(10, 6)):
    """
    Plot time in range by insulin dose.
    
    Parameters:
    -----------
    doses : array-like
        Insulin doses
    tir_values : array-like
        Corresponding time in range values
    low_threshold : float
        Lower bound of the target range
    high_threshold : float
        Upper bound of the target range
    optimal_dose : float, optional
        Optimal dose to highlight
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the data
    ax.plot(doses, tir_values, 'bo-', linewidth=2)
    
    # Mark optimal dose if provided
    if optimal_dose is not None:
        # Find closest dose and corresponding TIR
        closest_idx = np.abs(np.array(doses) - optimal_dose).argmin()
        optimal_tir = tir_values[closest_idx]
        
        # Add marker for optimal dose
        ax.plot([optimal_dose], [optimal_tir], 'r*', markersize=15, label=f'Optimal Dose: {optimal_dose:.2f} units')
        
        # Add annotation
        ax.annotate(f'Optimal: {optimal_dose:.2f} units\nTIR: {optimal_tir:.1f}%',
                   xy=(optimal_dose, optimal_tir),
                   xytext=(optimal_dose + 0.5, optimal_tir - 5),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Insulin Dose (units)')
    ax.set_ylabel('Time in Range (%)')
    ax.set_title(f'Time in Range ({low_threshold}-{high_threshold} mg/dL) by Insulin Dose')
    
    # Add grid
    ax.grid(True)
    
    # Add legend if optimal dose provided
    if optimal_dose is not None:
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig

def compare_tir_scenarios(scenarios, low_threshold=80, high_threshold=130, figsize=(12, 8)):
    """
    Compare time in range across different scenarios.
    
    Parameters:
    -----------
    scenarios : dict
        Dictionary mapping scenario names to glucose data
    low_threshold : float
        Lower bound of the target range
    high_threshold : float
        Upper bound of the target range
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the comparison
    """
    # Calculate metrics for each scenario
    metrics = {}
    for name, glucose_data in scenarios.items():
        metrics[name] = calculate_time_in_range(glucose_data, low_threshold, high_threshold)
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data for bar charts
    names = list(scenarios.keys())
    tir_values = [metrics[name]['tir'] for name in names]
    mean_values = [metrics[name]['mean_glucose'] for name in names]
    
    # Plot TIR comparison
    axs[0].bar(names, tir_values, color='green', alpha=0.7)
    axs[0].set_title('Time in Range Comparison')
    axs[0].set_ylabel('TIR (%)')
    axs[0].set_ylim(0, 100)
    
    # Add TIR values as text labels
    for i, v in enumerate(tir_values):
        axs[0].text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Plot mean glucose comparison
    axs[1].bar(names, mean_values, color='blue', alpha=0.7)
    axs[1].set_title('Mean Glucose Comparison')
    axs[1].set_ylabel('Mean Glucose (mg/dL)')
    
    # Add threshold lines
    axs[1].axhline(y=low_threshold, color='orange', linestyle='--', alpha=0.7, label='Range Threshold')
    axs[1].axhline(y=high_threshold, color='orange', linestyle='--', alpha=0.7)
    
    # Add mean values as text labels
    for i, v in enumerate(mean_values):
        axs[1].text(i, v + 2, f'{v:.1f}', ha='center')
    
    axs[1].legend()
    
    # Add overall title
    plt.suptitle(f'Comparison of Time in Range ({low_threshold}-{high_threshold} mg/dL) Across Scenarios')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    return fig