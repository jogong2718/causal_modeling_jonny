import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_counterfactual_datasets(base_dir, counterfactual_type):
    """
    Load all counterfactual datasets from the specified directory.
    
    Args:
        base_dir: Base directory for synthetic data
        counterfactual_type: 'dose_counterfactuals' or 'timing_counterfactuals'
    
    Returns:
        Dictionary of dataframes with factor/shift as keys
    """
    data_dir = os.path.join(base_dir, "data", counterfactual_type)
    datasets = {}
    
    # Get all CSV files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.endswith('_metrics.csv')]
    
    for file in files:
        filepath = os.path.join(data_dir, file)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Extract factor/shift from filename
        if counterfactual_type == 'dose_counterfactuals':
            # Extract factor (e.g., 0_8, 1_0, 1_2)
            factor = file.split('_')[2].replace('_', '.')
            key = factor
        else:  # timing_counterfactuals
            # Extract shift (e.g., minus60, plus30)
            if 'minus' in file:
                shift = f"-{file.split('minus')[1].split('_')[0]}"
            else:
                shift = file.split('plus')[1].split('_')[0]
            key = shift
        
        datasets[key] = df
    
    return datasets

def calculate_metrics(datasets, baseline_key):
    """
    Calculate MSE and MAE for each dataset compared to baseline.
    
    Args:
        datasets: Dictionary of dataframes
        baseline_key: Key for the baseline dataset
    
    Returns:
        DataFrame with metrics
    """
    print(f"Looking up baseline key: {baseline_key}")
    print(f"Available dataset keys: {datasets.keys()}")

    # Ensure baseline_key is formatted correctly
    if baseline_key not in datasets:
        corrected_key = str(int(float(baseline_key)))  # Convert "1.0" -> "1"
        print(f"Trying corrected key: {corrected_key}")
        baseline_key = corrected_key if corrected_key in datasets else baseline_key

    baseline = datasets.get(baseline_key)

    if baseline is None:
        raise KeyError(f"Baseline dataset not found for key: {baseline_key}")

    print(f"Successfully retrieved baseline dataset with key: {baseline_key}")


    baseline_key = str(int(float(baseline_key)))
    metrics = []
    
    for key, df in datasets.items():
        # Skip baseline comparison with itself
        if key == baseline_key:
            continue
        
        # Calculate metrics
        mse = mean_squared_error(baseline['glucose'], df['glucose'])
        mae = mean_absolute_error(baseline['glucose'], df['glucose'])
        rmse = np.sqrt(mse)
        
        # Calculate time in range metrics
        baseline_in_range = (baseline['glucose'].between(70, 180)).mean() * 100
        df_in_range = (df['glucose'].between(70, 180)).mean() * 100
        
        metrics.append({
            'scenario': key,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mean_glucose_diff': df['glucose'].mean() - baseline['glucose'].mean(),
            'max_glucose_diff': df['glucose'].max() - baseline['glucose'].max(),
            'min_glucose_diff': df['glucose'].min() - baseline['glucose'].min(),
            'time_in_range': df_in_range,
            'time_in_range_diff': df_in_range - baseline_in_range
        })
    
    # Create DataFrame and sort by scenario
    metrics_df = pd.DataFrame(metrics)
    
    if 'dose_counterfactuals' in baseline_key:
        # For dose counterfactuals, convert to float and sort numerically
        metrics_df['scenario_float'] = metrics_df['scenario'].astype(float)
        metrics_df = metrics_df.sort_values('scenario_float')
        metrics_df = metrics_df.drop('scenario_float', axis=1)
    else:
        # For timing counterfactuals, convert to int and sort numerically
        metrics_df['scenario_int'] = metrics_df['scenario'].astype(int)
        metrics_df = metrics_df.sort_values('scenario_int')
        metrics_df = metrics_df.drop('scenario_int', axis=1)
    
    return metrics_df

def plot_metrics(metrics_df, title, counterfactual_type):
    """
    Create visualizations for the metrics.
    
    Args:
        metrics_df: DataFrame with metrics
        title: Title for the plots
        counterfactual_type: 'dose' or 'timing'
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Mean Squared Error", "Mean Absolute Error", 
                        "Mean Glucose Difference", "Time in Range Difference"),
        vertical_spacing=0.15
    )
    
    # Define x-axis labels
    x_labels = metrics_df['scenario'].tolist()
    
    # Add MSE bars
    fig.add_trace(
        go.Bar(x=x_labels, y=metrics_df['mse'], name='MSE'),
        row=1, col=1
    )
    
    # Add MAE bars
    fig.add_trace(
        go.Bar(x=x_labels, y=metrics_df['mae'], name='MAE'),
        row=1, col=2
    )
    
    # Add mean glucose difference
    fig.add_trace(
        go.Bar(
            x=x_labels, 
            y=metrics_df['mean_glucose_diff'], 
            name='Difference',
            marker_color=['red' if x < 0 else 'green' for x in metrics_df['mean_glucose_diff']]
        ),
        row=2, col=1
    )
    
    # Add time in range difference
    fig.add_trace(
        go.Bar(
            x=x_labels, 
            y=metrics_df['time_in_range_diff'], 
            name='Difference',
            marker_color=['red' if x < 0 else 'green' for x in metrics_df['time_in_range_diff']]
        ),
        row=2, col=2
    )
    
    # Update x-axis titles
    if counterfactual_type == 'dose':
        x_title = "Insulin Dose Factor"
    else:  # timing
        x_title = "Insulin Timing Shift (minutes)"
        
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text=x_title, row=i, col=j)
    
    # Update y-axis titles
    fig.update_yaxes(title_text="MSE", row=1, col=1)
    fig.update_yaxes(title_text="MAE (mg/dL)", row=1, col=2)
    fig.update_yaxes(title_text="Mean Glucose Diff (mg/dL)", row=2, col=1)
    fig.update_yaxes(title_text="Time in Range Diff (%)", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        width=1000,
        showlegend=False
    )
    
    return fig

def evaluate_counterfactuals(base_dir="./synthetic_data"):
    """
    Main function to evaluate all counterfactual datasets.
    """
    # Create output directory for evaluations
    eval_dir = os.path.join(base_dir, "evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Evaluate dose counterfactuals
    print("Evaluating insulin dose counterfactuals...")
    dose_datasets = load_counterfactual_datasets(base_dir, "dose_counterfactuals")
    dose_metrics = calculate_metrics(dose_datasets, baseline_key="1")
    
    # Save metrics to CSV
    dose_metrics.to_csv(os.path.join(eval_dir, "dose_counterfactual_evaluation.csv"), index=False)
    
    # Create visualization
    dose_fig = plot_metrics(
        dose_metrics, 
        "Insulin Dose Counterfactual Evaluation", 
        "dose"
    )
    dose_fig.write_html(os.path.join(eval_dir, "dose_counterfactual_evaluation.html"))
    
    # Evaluate timing counterfactuals
    print("Evaluating insulin timing counterfactuals...")
    timing_datasets = load_counterfactual_datasets(base_dir, "timing_counterfactuals")
    timing_metrics = calculate_metrics(timing_datasets, baseline_key="0")
    
    # Save metrics to CSV
    timing_metrics.to_csv(os.path.join(eval_dir, "timing_counterfactual_evaluation.csv"), index=False)
    
    # Create visualization
    timing_fig = plot_metrics(
        timing_metrics, 
        "Insulin Timing Counterfactual Evaluation", 
        "timing"
    )
    timing_fig.write_html(os.path.join(eval_dir, "timing_counterfactual_evaluation.html"))
    
    # Generate additional pair plots for deeper analysis
    print("Generating pair plots of metrics...")
    
    # Combine metrics for comprehensive visualization
    dose_metrics['type'] = 'Dose'
    timing_metrics['type'] = 'Timing'
    all_metrics = pd.concat([dose_metrics, timing_metrics])
    
    # Create detailed metrics
    detailed_analysis = analyze_detailed_metrics(dose_datasets, timing_datasets)
    detailed_analysis.to_csv(os.path.join(eval_dir, "detailed_counterfactual_metrics.csv"), index=False)
    
    return {
        'dose_metrics': dose_metrics,
        'timing_metrics': timing_metrics,
        'dose_fig': dose_fig,
        'timing_fig': timing_fig
    }

def analyze_detailed_metrics(dose_datasets, timing_datasets):
    """
    Generate more detailed metrics for specific time periods like post-meal responses.
    """
    detailed_metrics = []
    
    # Analyze dose counterfactuals
    baseline_dose = dose_datasets["1"]
    for factor, df in dose_datasets.items():
        if factor == "1":
            continue
            
        # Find meal times from baseline
        meal_times = baseline_dose[baseline_dose['carbs'] > 0].index
        
        for meal_time in meal_times:
            # Look at 3-hour post-meal window
            end_time = meal_time + pd.Timedelta(hours=3)
            
            if end_time <= baseline_dose.index[-1]:
                meal_window_baseline = baseline_dose[meal_time:end_time]
                meal_window_comparison = df[meal_time:end_time]
                
                # Calculate metrics for this meal
                post_meal_mae = mean_absolute_error(
                    meal_window_baseline['glucose'], 
                    meal_window_comparison['glucose']
                )
                
                max_glucose_baseline = meal_window_baseline['glucose'].max()
                max_glucose_comparison = meal_window_comparison['glucose'].max()
                
                detailed_metrics.append({
                    'scenario_type': 'dose',
                    'scenario_value': factor,
                    'meal_datetime': meal_time,
                    'meal_size': baseline_dose.loc[meal_time, 'carbs'],
                    'insulin_dose': baseline_dose[baseline_dose.index <= meal_time]['insulin'].iloc[-1] if any(baseline_dose[baseline_dose.index <= meal_time]['insulin']) else 0,
                    'post_meal_mae': post_meal_mae,
                    'peak_glucose_diff': max_glucose_comparison - max_glucose_baseline,
                    'average_glucose_diff': meal_window_comparison['glucose'].mean() - meal_window_baseline['glucose'].mean()
                })
    
    # Analyze timing counterfactuals
    baseline_timing = timing_datasets["0"]
    for shift, df in timing_datasets.items():
        if shift == "0":
            continue
            
        # Find meal times from baseline
        meal_times = baseline_timing[baseline_timing['carbs'] > 0].index
        
        for meal_time in meal_times:
            # Look at 3-hour post-meal window
            end_time = meal_time + pd.Timedelta(hours=3)
            
            if end_time <= baseline_timing.index[-1]:
                meal_window_baseline = baseline_timing[meal_time:end_time]
                meal_window_comparison = df[meal_time:end_time]
                
                # Calculate metrics for this meal
                post_meal_mae = mean_absolute_error(
                    meal_window_baseline['glucose'], 
                    meal_window_comparison['glucose']
                )
                
                max_glucose_baseline = meal_window_baseline['glucose'].max()
                max_glucose_comparison = meal_window_comparison['glucose'].max()
                
                detailed_metrics.append({
                    'scenario_type': 'timing',
                    'scenario_value': shift,
                    'meal_datetime': meal_time,
                    'meal_size': baseline_timing.loc[meal_time, 'carbs'],
                    'insulin_dose': baseline_timing[baseline_timing.index <= meal_time]['insulin'].iloc[-1] if any(baseline_timing[baseline_timing.index <= meal_time]['insulin']) else 0,
                    'post_meal_mae': post_meal_mae,
                    'peak_glucose_diff': max_glucose_comparison - max_glucose_baseline,
                    'average_glucose_diff': meal_window_comparison['glucose'].mean() - meal_window_baseline['glucose'].mean()
                })
    
    return pd.DataFrame(detailed_metrics)

if __name__ == "__main__":
    results = evaluate_counterfactuals()
    
    print("\nEvaluation complete!")
    print("Files saved in the 'synthetic_data/evaluations' directory:")
    print("  - dose_counterfactual_evaluation.csv")
    print("  - dose_counterfactual_evaluation.html")
    print("  - timing_counterfactual_evaluation.csv")
    print("  - timing_counterfactual_evaluation.html")
    print("  - detailed_counterfactual_metrics.csv")
    
    print("\nSummary of dose counterfactual results:")
    print(results['dose_metrics'][['scenario', 'mae', 'mean_glucose_diff', 'time_in_range_diff']].to_string(index=False))
    
    print("\nSummary of timing counterfactual results:")
    print(results['timing_metrics'][['scenario', 'mae', 'mean_glucose_diff', 'time_in_range_diff']].to_string(index=False))