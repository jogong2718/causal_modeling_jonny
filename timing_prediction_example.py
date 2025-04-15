#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Glucose Prediction Example with ITS Models - Timing Analysis

This script demonstrates how to:
1. Train ITS models on glucose-insulin data
2. Save the trained models
3. Load models and use them for glucose predictions
4. Generate counterfactual predictions for different insulin timing
5. Find optimal insulin timing to reach target glucose levels
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", message=".*p-value may be inaccurate with fewer than 20 observations.*")
warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="Unknown keyword arguments", category=FutureWarning)

# Import the training and prediction modules
from its_package.training.model_trainer import ITSModelTrainer
from its_package.training.model_predictor import ITSPredictor
from its_package.data_handling.data_loader import load_csv_data
from its_package.data_handling.event_detection import detect_insulin_events

# Create output directory
output_dir = "output/timing_counterfactual_analysis"
os.makedirs(output_dir, exist_ok=True)
# Create models subdirectory
models_dir = os.path.join(output_dir, "models")
os.makedirs(models_dir, exist_ok=True)

def train_its_models():
    """Train ITS models on the synthetic glucose dataset"""
    print("=== Training ITS Models ===")
    
    # Load the data
    data_path = "synthetic_data/data/ml_dataset.csv"
    data = load_csv_data(data_path)
    print(f"Loaded data with shape: {data.shape}")
    
    # Detect insulin events for training
    events = detect_insulin_events(data, max_events=10)
    print(f"Detected {len(events)} insulin events for training")
    
    # Initialize model trainer
    trainer = ITSModelTrainer(output_dir=models_dir)
    
    # Train models on all events
    training_results = trainer.train_models(
        data=data,
        event_times=events,
        model_types=["causalimpact", "statsmodels"],
        pre_window="45min",
        post_window="120min",
        target_col="glucose",
        evaluate_prediction=True  # This enables the prediction-based evaluation
    )
    
    # Print training summary
    print("\nTraining Summary:")
    if training_results and isinstance(training_results, list) and len(training_results) > 0:
        print(f"Number of events processed: {len(training_results)}")
        if 'model_types' in training_results[0]:
            print(f"Model types: {', '.join(training_results[0]['model_types'])}")
        else:
            print("Model types: causalimpact, statsmodels")
    else:
        print("No training results available")
        print("Model types: causalimpact, statsmodels")
    
    # Save trained models
    try:
        saved_paths = trainer.save_models(base_filename="glucose_its_models")
        
        print("\nTraining complete!")
        print("Saved models:")
        for model_type, path in saved_paths.items():
            print(f"  - {model_type}: {path}")
    except AttributeError as e:
        print(f"\nWarning: Error saving models: {e}")
        print("Continuing with prediction using in-memory models...")
        # Create a dictionary with the model objects instead of paths
        saved_paths = {
            "causalimpact": trainer.models.get("causalimpact"),
            "statsmodels": trainer.models.get("statsmodels")
        }
    
    # Display prediction-based evaluation results
    if "causalimpact_prediction" in trainer.evaluation_results:
        ci_pred_metrics = trainer.evaluation_results["causalimpact_prediction"]
        print("\nCausalImpact Prediction-based Evaluation:")
        print(f"  Average MAE: {ci_pred_metrics['MAE'].mean():.2f}")
        print(f"  Average RMSE: {ci_pred_metrics['RMSE'].mean():.2f}")
        print(f"  Average R²: {ci_pred_metrics['R2'].mean():.2f}")
    
    if "statsmodels_prediction" in trainer.evaluation_results:
        sm_pred_metrics = trainer.evaluation_results["statsmodels_prediction"]
        print("\nStatsModels Prediction-based Evaluation:")
        print(f"  Average MAE: {sm_pred_metrics['MAE'].mean():.2f}")
        print(f"  Average RMSE: {sm_pred_metrics['RMSE'].mean():.2f}")
        print(f"  Average R²: {sm_pred_metrics['R2'].mean():.2f}")
    
    return saved_paths

def make_predictions(model_paths):
    """Make glucose predictions using trained models"""
    print("\n=== Making Glucose Predictions ===")
    
    # Load data for prediction
    data_path = "synthetic_data/data/ml_dataset.csv"
    data = load_csv_data(data_path)
    
    # Select an insulin event for prediction
    events = detect_insulin_events(data, max_events=3)
    prediction_event = events[0]
    print(f"Selected event at {prediction_event} for prediction")
    
    # Extract pre-period data for this event
    pre_window = "45min"
    pre_start = prediction_event - pd.Timedelta(pre_window)
    pre_period_data = data.loc[pre_start:prediction_event].copy()
    
    # Get the actual insulin dose at the event
    actual_dose = data.loc[prediction_event, "insulin"]
    print(f"Actual insulin dose: {actual_dose:.2f}u")
    
    # Initialize predictor and load models
    predictor = ITSPredictor()
    
    # Check if we have model paths or in-memory models
    if isinstance(model_paths.get("causalimpact"), str):
        # Load CausalImpact model from file
        if "causalimpact" in model_paths:
            predictor.load_model(model_paths["causalimpact"], "causalimpact")
    else:
        # Use in-memory CausalImpact model
        if "causalimpact" in model_paths and model_paths["causalimpact"] is not None:
            predictor.models["causalimpact"] = model_paths["causalimpact"]
            # Initialize model_info for causalimpact
            predictor.model_info["causalimpact"] = {
                "pre_period": "45min",
                "post_period": "120min",
                "target_col": "glucose",
                "insulin_dose": actual_dose
            }
            print("Using in-memory CausalImpact model")
    
    if isinstance(model_paths.get("statsmodels"), str):
        # Load StatsModels ITS model from file
        if "statsmodels" in model_paths:
            predictor.load_model(model_paths["statsmodels"], "statsmodels")
    else:
        # Use in-memory StatsModels model
        if "statsmodels" in model_paths and model_paths["statsmodels"] is not None:
            predictor.models["statsmodels"] = model_paths["statsmodels"]
            # Initialize model_info for statsmodels
            predictor.model_info["statsmodels"] = {
                "pre_period": "45min",
                "post_period": "120min",
                "target_col": "glucose",
                "insulin_dose": actual_dose
            }
            print("Using in-memory StatsModels model")
    
    # Make predictions with different model types
    model_types = []
    if "causalimpact" in predictor.models:
        model_types.append("causalimpact")
    if "statsmodels" in predictor.models:
        model_types.append("statsmodels")
    if len(model_types) > 1:
        model_types.append("ensemble")
    
    # Extract post-period data for validation
    post_window = "2h"
    post_end = prediction_event + pd.Timedelta(post_window)
    actual_post_data = data.loc[prediction_event:post_end].copy()
    
    prediction_results = {}
    
    for model_type in model_types:
        print(f"\nPredicting with {model_type} model...")
        
        # Predict glucose after the event
        predictions = predictor.predict_glucose(
            pre_period_data=pre_period_data,
            intervention_time=prediction_event,
            post_period_length="2h",
            intervention_value=actual_dose,
            model_type=model_type,
            time_frequency="5min"
        )
        
        # Store predictions for later comparison
        prediction_results[model_type] = predictions
        
        # Plot the prediction
        output_path = os.path.join(output_dir, f"{model_type}_prediction.png")
        predictor.plot_prediction(
            pre_period_data=pre_period_data,
            predictions=predictions,
            intervention_time=prediction_event,
            target_col="glucose",
            output_path=output_path
        )
        
        print(f"Prediction plot saved to {output_path}")
    
    # Compare predictions with actual post-period data
    compare_with_actual(prediction_results, actual_post_data, prediction_event)
    
    return predictor, prediction_event, pre_period_data, actual_dose

def compare_with_actual(prediction_results, actual_data, event_time):
    """Compare predictions with actual data and calculate metrics"""
    print("\n=== Comparing Predictions with Actual Data ===")
    
    # Create a figure to plot actual vs predicted
    plt.figure(figsize=(12, 8))
    
    # Plot actual data
    plt.plot(actual_data.index, actual_data['glucose'], 'k-', linewidth=2, label="Actual")
    
    # Plot predictions from each model
    colors = {'causalimpact': 'red', 'statsmodels': 'green', 'ensemble': 'purple'}
    prediction_metrics = {}
    
    for model_type, predictions in prediction_results.items():
        # Plot predictions
        color = colors.get(model_type, 'blue')
        plt.plot(predictions.index, predictions['predicted'], 
                 color=color, linestyle='--', 
                 linewidth=1.5, label=f"{model_type.capitalize()} Prediction")
        
        # Calculate error metrics for overlapping time points
        common_times = set(actual_data.index) & set(predictions.index)
        if common_times:
            common_idx = sorted(list(common_times))
            actual_values = actual_data.loc[common_idx, 'glucose']
            predicted_values = predictions.loc[common_idx, 'predicted']
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from math import sqrt
            
            mae = mean_absolute_error(actual_values, predicted_values)
            rmse = sqrt(mean_squared_error(actual_values, predicted_values))
            r2 = r2_score(actual_values, predicted_values)
            
            prediction_metrics[model_type] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2
            }
            
            print(f"{model_type.capitalize()} prediction metrics:")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.2f}")
    
    # Mark intervention time
    plt.axvline(x=event_time, color='k', linestyle='--', label='Intervention')
    
    # Add labels and title
    plt.title("Prediction Comparison with Actual Glucose Data")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(output_dir, "prediction_comparison.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Comparison plot saved to {output_path}")
    
    # Create a bar chart to compare prediction metrics
    if prediction_metrics:
        metrics_to_plot = ['MAE', 'RMSE']
        plt.figure(figsize=(10, 6))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics[metric] for model, metrics in prediction_metrics.items()]
            model_names = list(prediction_metrics.keys())
            x = np.arange(len(model_names))
            
            plt.subplot(1, len(metrics_to_plot), i+1)
            bars = plt.bar(x, values, width=0.6, alpha=0.7)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.title(f'Prediction {metric}')
            plt.ylabel(metric)
            plt.xticks(x, [m.capitalize() for m in model_names])
            plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "prediction_metrics_comparison.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Metrics comparison chart saved to {output_path}")
    
    return prediction_metrics

def explore_counterfactual_timing(predictor, event_time, pre_period_data, actual_dose):
    """Explore counterfactual scenarios with different insulin timing"""
    print("\n=== Exploring Counterfactual Insulin Timing ===")
    
    # Define counterfactual timing offsets (in minutes)
    timing_offsets = [-30, -15, 0, 15, 30, 60]
    print(f"Exploring timing offsets: {timing_offsets} minutes")
    
    # Predict glucose for different timing
    counterfactual_results = {}
    
    for offset in timing_offsets:
        # Calculate the counterfactual intervention time
        cf_intervention_time = event_time + pd.Timedelta(minutes=offset)
        
        # Skip if the counterfactual time is before the pre-period start
        if cf_intervention_time < pre_period_data.index[0]:
            print(f"Skipping offset {offset} minutes (before pre-period)")
            continue
        
        # Predict glucose for this timing
        cf_predictions = predictor.predict_glucose(
            pre_period_data=pre_period_data,
            intervention_time=cf_intervention_time,
            post_period_length="2h",
            intervention_value=actual_dose,
            model_type="ensemble",
            time_frequency="5min"
        )
        
        # Store predictions
        counterfactual_results[offset] = {
            'intervention_time': cf_intervention_time,
            'predictions': cf_predictions
        }
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    # Plot pre-period data
    plt.plot(pre_period_data.index, pre_period_data['glucose'], 'k-', 
             linewidth=1.5, label="Pre-intervention")
    
    # Plot predictions for each timing
    colors = plt.cm.viridis(np.linspace(0, 1, len(counterfactual_results)))
    for i, (offset, result) in enumerate(counterfactual_results.items()):
        cf_time = result['intervention_time']
        predictions = result['predictions']
        
        # Plot the prediction
        plt.plot(predictions.index, predictions['predicted'], 
                 color=colors[i], linestyle='--', 
                 linewidth=1.5, label=f"{offset} min offset")
        
        # Mark the intervention time
        plt.axvline(x=cf_time, color=colors[i], linestyle=':', alpha=0.5)
    
    # Mark the original intervention time
    plt.axvline(x=event_time, color='k', linestyle='--', label='Original Intervention')
    
    # Add labels and title
    plt.title("Counterfactual Timing Comparison")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(output_dir, "counterfactual_timing_comparison.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Counterfactual timing comparison plot saved to {output_path}")
    
    # Define target range for glucose (healthy range)
    low_threshold = 70  # mg/dL - lower bound for healthy glucose
    high_threshold = 180  # mg/dL - upper bound for healthy glucose
    
    print(f"\nFinding optimal timing to maximize time in range ({low_threshold}-{high_threshold} mg/dL)")
    
    # Find optimal timing
    optimal_result = find_optimal_timing(
        predictor=predictor,
        pre_period_data=pre_period_data,
        original_event_time=event_time,
        timing_range=(-60, 60),  # minutes
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        post_period="2h",
        model_type="ensemble",
        n_steps=20
    )
    
    # Plot time in range comparison for different timing
    plot_timing_tir_comparison(
        counterfactual_results=counterfactual_results,
        original_event_time=event_time,
        pre_period_data=pre_period_data,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        output_dir=output_dir
    )
    
    return optimal_result

def find_optimal_timing(predictor, pre_period_data, original_event_time, timing_range, 
                       low_threshold=70, high_threshold=180, post_period="2h",
                       model_type="ensemble", n_steps=20):
    """
    Find the optimal timing for insulin administration to maximize time in range
    
    Parameters:
    -----------
    predictor : ITSPredictor
        The predictor object with loaded models
    pre_period_data : DataFrame
        Data for the pre-intervention period
    original_event_time : datetime
        The original intervention time
    timing_range : tuple
        (min_offset, max_offset) in minutes
    low_threshold : float
        Lower bound for target glucose range
    high_threshold : float
        Upper bound for target glucose range
    post_period : str
        Length of post-period to analyze
    model_type : str
        Type of model to use
    n_steps : int
        Number of timing offsets to try
        
    Returns:
    --------
    dict
        Dictionary with optimal timing results
    """
    print(f"Finding optimal timing between {timing_range[0]} and {timing_range[1]} minutes offset")
    
    # Generate timing offsets to try
    timing_offsets = np.linspace(timing_range[0], timing_range[1], n_steps)
    
    # Get the insulin dose from the first available model
    if model_type == "ensemble":
        # Try to get insulin dose from causalimpact first, then statsmodels
        insulin_dose = None
        if "causalimpact" in predictor.model_info:
            insulin_dose = predictor.model_info["causalimpact"].get("insulin_dose", 1.0)
        elif "statsmodels" in predictor.model_info:
            insulin_dose = predictor.model_info["statsmodels"].get("insulin_dose", 1.0)
        else:
            insulin_dose = 1.0  # Default value if no models available
    else:
        insulin_dose = predictor.model_info[model_type].get("insulin_dose", 1.0)
    
    # Store results for each timing
    results = []
    
    for offset in timing_offsets:
        # Calculate the counterfactual intervention time
        cf_intervention_time = original_event_time + pd.Timedelta(minutes=offset)
        
        # Skip if the counterfactual time is before the pre-period start
        if cf_intervention_time < pre_period_data.index[0]:
            print(f"Skipping offset {offset:.1f} minutes (before pre-period)")
            continue
        
        # Predict glucose for this timing
        predictions = predictor.predict_glucose(
            pre_period_data=pre_period_data,
            intervention_time=cf_intervention_time,
            post_period_length=post_period,
            intervention_value=insulin_dose,
            model_type=model_type,
            time_frequency="5min"
        )
        
        # Calculate time in range metrics
        glucose_values = predictions['predicted']
        below_range = (glucose_values < low_threshold).mean() * 100
        above_range = (glucose_values > high_threshold).mean() * 100
        in_range = 100 - below_range - above_range
        
        # Calculate mean glucose
        mean_glucose = glucose_values.mean()
        
        # Store results
        results.append({
            'offset': offset,
            'intervention_time': cf_intervention_time,
            'below_range': below_range,
            'in_range': in_range,
            'above_range': above_range,
            'mean_glucose': mean_glucose
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Find the timing with maximum time in range
    optimal_idx = results_df['in_range'].idxmax()
    optimal_result = results_df.loc[optimal_idx].to_dict()
    optimal_result['optimal_offset'] = optimal_result['offset']
    optimal_result['optimal_timing'] = optimal_result['intervention_time']
    optimal_result['low_threshold'] = low_threshold
    optimal_result['high_threshold'] = high_threshold
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot time in range metrics
    plt.subplot(2, 1, 1)
    plt.plot(results_df['offset'], results_df['below_range'], 'r-', label='Below Range')
    plt.plot(results_df['offset'], results_df['in_range'], 'g-', label='In Range')
    plt.plot(results_df['offset'], results_df['above_range'], 'b-', label='Above Range')
    plt.axvline(x=optimal_result['offset'], color='k', linestyle='--', label='Optimal Timing')
    plt.title('Time in Range by Timing Offset')
    plt.xlabel('Timing Offset (minutes)')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot mean glucose
    plt.subplot(2, 1, 2)
    plt.plot(results_df['offset'], results_df['mean_glucose'], 'k-')
    plt.axvline(x=optimal_result['offset'], color='k', linestyle='--', label='Optimal Timing')
    plt.axhline(y=low_threshold, color='r', linestyle=':', label='Lower Threshold')
    plt.axhline(y=high_threshold, color='b', linestyle=':', label='Upper Threshold')
    plt.title('Mean Glucose by Timing Offset')
    plt.xlabel('Timing Offset (minutes)')
    plt.ylabel('Mean Glucose (mg/dL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, "optimal_timing_tir.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Time-in-range optimization plot saved to {output_path}")
    
    return optimal_result

def plot_timing_tir_comparison(counterfactual_results, original_event_time, pre_period_data,
                              low_threshold=70, high_threshold=180, output_dir=None):
    """
    Plot time in range comparison for different timing offsets
    
    Parameters:
    -----------
    counterfactual_results : dict
        Dictionary mapping timing offsets to prediction results
    original_event_time : datetime
        The original intervention time
    pre_period_data : DataFrame
        Data for the pre-intervention period
    low_threshold : float
        Lower bound for target glucose range
    high_threshold : float
        Upper bound for target glucose range
    output_dir : str
        Directory to save the plot
    """
    # Calculate time in range metrics for each timing
    tir_results = []
    
    for offset, result in counterfactual_results.items():
        predictions = result['predictions']
        glucose_values = predictions['predicted']
        
        below_range = (glucose_values < low_threshold).mean() * 100
        above_range = (glucose_values > high_threshold).mean() * 100
        in_range = 100 - below_range - above_range
        
        mean_glucose = glucose_values.mean()
        
        tir_results.append({
            'offset': offset,
            'below_range': below_range,
            'in_range': in_range,
            'above_range': above_range,
            'mean_glucose': mean_glucose
        })
    
    # Convert to DataFrame
    tir_df = pd.DataFrame(tir_results)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot time in range metrics
    plt.subplot(2, 2, 1)
    plt.bar(tir_df['offset'], tir_df['below_range'], color='red', alpha=0.7, label='Below Range')
    plt.bar(tir_df['offset'], tir_df['in_range'], bottom=tir_df['below_range'], 
            color='green', alpha=0.7, label='In Range')
    plt.bar(tir_df['offset'], tir_df['above_range'], 
            bottom=tir_df['below_range'] + tir_df['in_range'], 
            color='blue', alpha=0.7, label='Above Range')
    plt.axvline(x=0, color='k', linestyle='--', label='Original Timing')
    plt.title('Time in Range by Timing Offset')
    plt.xlabel('Timing Offset (minutes)')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Plot mean glucose
    plt.subplot(2, 2, 2)
    plt.bar(tir_df['offset'], tir_df['mean_glucose'], color='purple', alpha=0.7)
    plt.axvline(x=0, color='k', linestyle='--', label='Original Timing')
    plt.axhline(y=low_threshold, color='r', linestyle=':', label='Lower Threshold')
    plt.axhline(y=high_threshold, color='b', linestyle=':', label='Upper Threshold')
    plt.title('Mean Glucose by Timing Offset')
    plt.xlabel('Timing Offset (minutes)')
    plt.ylabel('Mean Glucose (mg/dL)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Plot glucose trajectories
    plt.subplot(2, 2, (3, 4))
    
    # Plot pre-period data
    plt.plot(pre_period_data.index, pre_period_data['glucose'], 'k-', 
             linewidth=1.5, label="Pre-intervention")
    
    # Plot predictions for each timing
    colors = plt.cm.viridis(np.linspace(0, 1, len(counterfactual_results)))
    for i, (offset, result) in enumerate(counterfactual_results.items()):
        cf_time = result['intervention_time']
        predictions = result['predictions']
        
        # Plot the prediction
        plt.plot(predictions.index, predictions['predicted'], 
                 color=colors[i], linestyle='--', 
                 linewidth=1.5, label=f"{offset} min offset")
        
        # Mark the intervention time
        plt.axvline(x=cf_time, color=colors[i], linestyle=':', alpha=0.5)
    
    # Mark the original intervention time
    plt.axvline(x=original_event_time, color='k', linestyle='--', label='Original Intervention')
    
    # Add target range shading
    plt.axhspan(low_threshold, high_threshold, color='green', alpha=0.1, label='Target Range')
    
    # Add labels and title
    plt.title("Glucose Trajectories by Timing Offset")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    if output_dir:
        output_path = os.path.join(output_dir, "timing_tir_comparison.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Time in range comparison plot saved to {output_path}")
    else:
        plt.show()

def main():
    """Main function to run the example"""
    
    # Check if models already exist
    ci_model_path = os.path.join(models_dir, "glucose_its_models_causalimpact.pkl")
    sm_model_path = os.path.join(models_dir, "glucose_its_models_statsmodels.pkl")
    
    if os.path.exists(ci_model_path) and os.path.exists(sm_model_path):
        print("Using existing trained models")
        model_paths = {
            "causalimpact": ci_model_path,
            "statsmodels": sm_model_path
        }
    else:
        # Train models
        model_paths = train_its_models()
    
    # Make predictions with trained models
    predictor, event_time, pre_period_data, actual_dose = make_predictions(model_paths)
    
    # Explore counterfactual timing scenarios
    optimal_result = explore_counterfactual_timing(predictor, event_time, pre_period_data, actual_dose)
    
    print("\n=== Summary ===")
    print(f"For optimal Time In Range ({optimal_result['in_range']:.1f}%):")
    print(f"Target glucose range: {optimal_result.get('low_threshold', 70)}-{optimal_result.get('high_threshold', 180)} mg/dL")
    print(f"Recommended timing offset: {optimal_result['optimal_offset']:.1f} minutes")
    print(f"Expected time in range: {optimal_result['in_range']:.1f}%")
    print(f"Expected mean glucose: {optimal_result['mean_glucose']:.1f} mg/dL")
    print(f"Below range: {optimal_result['below_range']:.1f}%, Above range: {optimal_result['above_range']:.1f}%")
    print(f"Original timing: {event_time}")
    print(f"Optimal timing: {optimal_result['optimal_timing']}")
    
    print("\nExample complete! All outputs saved to:", output_dir)

if __name__ == "__main__":
    main()