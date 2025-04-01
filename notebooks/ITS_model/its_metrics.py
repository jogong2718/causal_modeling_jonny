import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import sklearn.metrics as metrics
import pickle
import os

# Create metrics directory if it doesn't exist
if not os.path.exists("data_metrics"):
    os.makedirs("data_metrics")

def evaluate_models():
    """
    Evaluate the causal impact models using various metrics
    """
    print("\n" + "="*50)
    print("EVALUATING CAUSAL IMPACT MODELS")
    print("="*50)
    
    # Load results from modeling
    try:
        with open('model_results.pkl', 'rb') as f:
            results = pickle.load(f)
            
        filtered_events = results['filtered_events']
        summary_list = results['summary_list']
        model_dict = results['model_dict']
    except FileNotFoundError:
        print("Error: model_results.pkl not found. Run causal_impact_modeling.py first.")
        return
    except Exception as e:
        print(f"Error loading model results: {e}")
        return
    
    # Initialize a dictionary to store metrics for each event
    all_metrics = {}
    
    if len(filtered_events) > 0 and len(summary_list) > 0:
        for event_idx, event_key in enumerate([f"event_{i}" for i in range(len(model_dict))]):
            if event_key not in model_dict:
                print(f"Skipping {event_key} - not found in model dictionary")
                continue
                
            try:
                # Extract model and data from dictionary
                event_model = model_dict[event_key]
                impact = event_model['impact']
                event_details = event_model['event_details']
                
                # Extract actual and predicted values for the post-period
                post_period = event_details['post_period']
                post_data = impact.inferences.loc[impact.inferences.index >= post_period[0]]
                actual = post_data['response']
                predicted = post_data['point_pred']
                
                # Calculate metrics
                event_metrics = {
                    'event_time': event_details['event_time'],
                    'MAE': metrics.mean_absolute_error(actual, predicted),
                    'MSE': metrics.mean_squared_error(actual, predicted),
                    'RMSE': sqrt(metrics.mean_squared_error(actual, predicted)),
                    'R2': metrics.r2_score(actual, predicted),
                    'MAPE': np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100  # Added small constant to avoid division by zero
                }
                
                # Add to the overall metrics dictionary
                all_metrics[event_key] = event_metrics
                
                print(f"\nMetrics for {event_key} at {event_details['event_time']}:")
                print(f"MAE: {event_metrics['MAE']:.4f}")
                print(f"MSE: {event_metrics['MSE']:.4f}")
                print(f"RMSE: {event_metrics['RMSE']:.4f}")
                print(f"R²: {event_metrics['R2']:.4f}")
                print(f"MAPE: {event_metrics['MAPE']:.2f}%")
                
                # Create a residual plot
                plt.figure(figsize=(10, 6))
                residuals = actual - predicted
                plt.scatter(predicted, residuals)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                plt.title(f'Residual Plot for {event_key}')
                plt.savefig(f"data_metrics/{event_key}_residuals.png")
                plt.close()
                
                # Create a prediction vs actual plot
                plt.figure(figsize=(10, 6))
                plt.plot(post_data.index, actual, 'b-', label='Actual')
                plt.plot(post_data.index, predicted, 'r--', label='Predicted')
                plt.fill_between(
                    post_data.index,
                    post_data['point_pred_lower'],
                    post_data['point_pred_upper'],
                    color='r', alpha=0.2
                )
                plt.xlabel('Time')
                plt.ylabel('Glucose Level')
                plt.title(f'Actual vs Predicted for {event_key}')
                plt.legend()
                plt.savefig(f"data_metrics/{event_key}_prediction.png")
                plt.close()
                
            except Exception as e:
                print(f"Error calculating metrics for {event_key}: {e}")
        
        # Create a summary DataFrame of all metrics
        metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
        
        # Calculate average metrics across all events
        avg_metrics = {
            'MAE': metrics_df['MAE'].mean(),
            'MSE': metrics_df['MSE'].mean(),
            'RMSE': metrics_df['RMSE'].mean(),
            'R2': metrics_df['R2'].mean(),
            'MAPE': metrics_df['MAPE'].mean()
        }
        
        print("\n" + "="*50)
        print("AVERAGE METRICS ACROSS ALL EVENTS:")
        print(f"Average MAE: {avg_metrics['MAE']:.4f}")
        print(f"Average MSE: {avg_metrics['MSE']:.4f}")
        print(f"Average RMSE: {avg_metrics['RMSE']:.4f}")
        print(f"Average R²: {avg_metrics['R2']:.4f}")
        print(f"Average MAPE: {avg_metrics['MAPE']:.2f}%")
        
        # Save the metrics to CSV
        metrics_df.to_csv("data_metrics/all_event_metrics.csv")
        
        # Create a bar chart comparing metrics across events
        plt.figure(figsize=(14, 10))
        
        metrics_to_plot = ['MAE', 'RMSE', 'MAPE']
        colors = ['blue', 'green', 'orange']
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(len(metrics_to_plot), 1, i+1)
            plt.bar(metrics_df.index, metrics_df[metric], color=colors[i])
            plt.axhline(y=avg_metrics[metric], color='r', linestyle='--', label=f'Avg {metric}')
            plt.title(f'{metric} by Event')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig("data_metrics/metrics_comparison.png")
        plt.close()
        
        # Create a correlation matrix to see relationships between metrics
        if len(metrics_df) > 1:  # Only if we have more than one event
            corr_matrix = metrics_df.corr()
            plt.figure(figsize=(8, 8))
            plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.title('Correlation Matrix of Evaluation Metrics')
            plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
            plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
            
            # Add correlation values to the heatmap
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center', color='black')
            
            plt.tight_layout()
            plt.savefig("data_metrics/metrics_correlation.png")
            plt.close()
    else:
        print("No events were successfully analyzed. Cannot calculate metrics.")
    
    return all_metrics

if __name__ == "__main__":
    evaluate_models()
    print("\nEvaluation complete. Results saved to data_metrics/ directory.")