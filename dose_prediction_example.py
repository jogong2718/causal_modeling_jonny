#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Glucose Prediction Example with ITS Models

This script demonstrates how to:
1. Train ITS models on glucose-insulin data
2. Save the trained models
3. Load models and use them for glucose predictions
4. Generate counterfactual predictions for different insulin doses
5. Find optimal insulin doses to reach target glucose levels
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
output_dir = "output/dose_counterfactual_analysis"
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
        pre_window="43200min",
        post_window="240min",
        target_col="glucose",
        evaluate_prediction=True  # This enables the prediction-based evaluation
    )
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Number of events processed: {len(training_results)}")
    # Check if training_results is a list and has model_types in the first item
    if training_results and isinstance(training_results, list) and training_results[0] and 'models_trained' in training_results[0]:
        model_types = list(training_results[0]['models_trained'].keys())
        print(f"Model types: {', '.join(model_types)}")
    elif training_results and isinstance(training_results, dict) and 'models_trained' in training_results:
        # Handle case where it might be a single dictionary
        model_types = list(training_results['models_trained'].keys())
        print(f"Model types: {', '.join(model_types)}")
    else:
        print("No model types information available in training results")
    
    # Save trained models
    saved_paths = trainer.save_models(base_filename="glucose_its_models")
    
    print("\nTraining complete!")
    print("Saved models:")
    for model_type, path in saved_paths.items():
        print(f"  - {model_type}: {path}")
    
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
    pre_window = "43200min"
    pre_start = prediction_event - pd.Timedelta(pre_window)
    pre_period_data = data.loc[pre_start:prediction_event].copy()
    
    # Get the actual insulin dose at the event
    actual_dose = data.loc[prediction_event, "insulin"]
    print(f"Actual insulin dose: {actual_dose:.2f}u")
    
    # Initialize predictor and load models
    predictor = ITSPredictor()
    
    # Load CausalImpact model
    if "causalimpact" in model_paths:
        predictor.load_model(model_paths["causalimpact"], "causalimpact")
        
    # Load StatsModels ITS model
    if "statsmodels" in model_paths:
        predictor.load_model(model_paths["statsmodels"], "statsmodels")
    
    # Make predictions with different model types
    model_types = []
    if "causalimpact" in model_paths:
        model_types.append("causalimpact")
    if "statsmodels" in model_paths:
        model_types.append("statsmodels")
    if len(model_types) > 1:
        model_types.append("ensemble")
    
    # Extract post-period data for validation
    post_window = "4h"
    post_end = prediction_event + pd.Timedelta(post_window)
    actual_post_data = data.loc[prediction_event:post_end].copy()
    
    prediction_results = {}
    
    for model_type in model_types:
        print(f"\nPredicting with {model_type} model...")
        
        # Predict glucose after the event
        predictions = predictor.predict_glucose(
            pre_period_data=pre_period_data,
            intervention_time=prediction_event,
            post_period_length="4h",
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
        # Plot predictions - FIX: separate the color and line style parameters
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

def explore_counterfactuals(predictor, event_time, pre_period_data, actual_dose):
    """Explore counterfactual scenarios with different insulin doses"""
    print("\n=== Exploring Counterfactual Insulin Doses ===")
    
    # Define counterfactual doses
    counterfactual_doses = [0.0, actual_dose/2, actual_dose*0.8, actual_dose*1.2, actual_dose*1.5, actual_dose*2]
    print(f"Exploring doses: {[round(d, 2) for d in counterfactual_doses]}")
    
    # Predict glucose for different doses
    counterfactual_results = predictor.predict_counterfactual_doses(
        pre_period_data=pre_period_data,
        intervention_time=event_time,
        actual_dose=actual_dose,
        counterfactual_doses=counterfactual_doses,
        post_period_length="4h",
        model_type="ensemble",
        time_frequency="5min"
    )
    
    # Plot comparison
    output_path = os.path.join(output_dir, "counterfactual_dose_comparison.png")
    predictor.plot_counterfactual_comparison(
        pre_period_data=pre_period_data,
        counterfactual_results=counterfactual_results,
        intervention_time=event_time,
        target_col="glucose",
        output_path=output_path
    )
    
    print(f"Counterfactual comparison plot saved to {output_path}")
    
    # Define target range for glucose (healthy range)
    low_threshold = 70  # mg/dL - lower bound for healthy glucose
    high_threshold = 180  # mg/dL - upper bound for healthy glucose
    
    print(f"\nFinding optimal dose to maximize time in range ({low_threshold}-{high_threshold} mg/dL)")
    
    # Use maximize_time_in_range instead of find_optimal_dose
    optimal_result = predictor.maximize_time_in_range(
        pre_period_data=pre_period_data,
        intervention_time=event_time,
        dose_range=(0.0, actual_dose*2),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        post_period="4h",
        model_type="ensemble",
        n_steps=20
    )
    
    # Plot time in range comparison for different doses
    output_path = os.path.join(output_dir, "tir_comparison.png")
    predictor.plot_time_in_range_comparison(
        pre_period_data=pre_period_data,
        counterfactual_results=counterfactual_results,
        intervention_time=event_time,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        output_path=output_path
    )
    print(f"Time in range comparison plot saved to {output_path}")
    
    # Save the TIR optimization plot
    output_path = os.path.join(output_dir, "optimal_tir_dose.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Time-in-range optimization plot saved to {output_path}")
    
    return optimal_result

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
    
    # Explore counterfactual scenarios
    optimal_result = explore_counterfactuals(predictor, event_time, pre_period_data, actual_dose)
    
    print("\n=== Summary ===")
    print(f"For optimal Time In Range ({optimal_result['below_range'] + optimal_result['optimal_tir']:.1f}%):")
    print(f"Target glucose range: {optimal_result.get('low_threshold', 70)}-{optimal_result.get('high_threshold', 180)} mg/dL")
    print(f"Recommended insulin dose: {optimal_result['optimal_dose']:.2f}u")
    print(f"Expected time in range: {optimal_result['optimal_tir']:.1f}%")
    print(f"Expected mean glucose: {optimal_result['mean_glucose']:.1f} mg/dL")
    print(f"Below range: {optimal_result['below_range']:.1f}%, Above range: {optimal_result['above_range']:.1f}%")
    print(f"Actual dose used: {actual_dose:.2f}u")
    
    print("\nExample complete! All outputs saved to:", output_dir)

if __name__ == "__main__":
    main()
