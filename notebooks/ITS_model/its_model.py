import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causalimpact import CausalImpact
import warnings
import os
import pickle

# Create necessary directories if they don't exist
for directory in ["data_visual", "data_causal", "data_metrics"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="DataFrame.fillna with 'method' is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame.applymap has been deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="No frequency information was provided", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", message="No frequency information was provided", category=UserWarning)
warnings.filterwarnings("ignore", message="Keyword arguments have been passed to the optimizer", category=FutureWarning)
warnings.filterwarnings("ignore", message="Unknown keyword arguments: dict_keys", category=FutureWarning)

def run_causal_modeling(data_filepath):
    """
    Run causal impact analysis on insulin intervention events
    
    Args:
        data_filepath: Path to the CSV data file
        
    Returns:
        tuple: (filtered_events, summary_list, model_dict) containing analysis results
    """
    # Load data
    data = pd.read_csv(data_filepath, index_col=0)
    data.index = pd.to_datetime(data.index)

    print("Data Shape:", data.shape)
    print("Date Range:", data.index.min(), "to", data.index.max())
    print("Columns in dataset:", data.columns.tolist())

    # Identify intervention events
    all_events = data.index[data['insulin'] > 0]
    filtered_events = []
    last_event = None

    for event in all_events:
        if last_event is None or (event - last_event).total_seconds() >= 3600:
            filtered_events.append(event)
            last_event = event

    print(f"\nFound {len(filtered_events)} filtered intervention times")
    filtered_events = filtered_events[:10]  # Limit to first 10 events for testing

    summary_list = []
    model_dict = {}  # Store CausalImpact models

    if len(filtered_events) == 0:
        print("No interventions found.")
    else:
        for event_idx, event in enumerate(filtered_events):
            print(f"\n{'='*50}")
            print(f"Processing event {event_idx+1}/{len(filtered_events)} at {event}")
            
            # Define the window around the event
            window_start = event - pd.Timedelta("45min")
            window_end = event + pd.Timedelta("30min")
            
            try:
                window_data = data.loc[window_start:window_end].copy()
                window_data = window_data.fillna(method='ffill')
                print(f"Window data shape: {window_data.shape}")
                
                closest_pre = window_data.index[window_data.index <= event].max()
                closest_post = window_data.index[window_data.index > event].min()
                
                if pd.isna(closest_pre) or pd.isna(closest_post):
                    print(f"Skipping event at {event} - cannot establish pre/post boundaries")
                    continue
                    
                pre_period = [window_data.index.min(), closest_pre]
                post_period = [closest_post, window_data.index.max()]
                
                ci_data = window_data.copy()
                ci_data = ci_data.replace([np.inf, -np.inf], np.nan).dropna()
                ci_data = ci_data.loc[:, ci_data.nunique() > 1]
                
                print(f"Pre-period: {pre_period[0]} to {pre_period[1]}")
                print(f"Post-period: {post_period[0]} to {post_period[1]}")
                
                # Diagnostic plot
                plt.figure(figsize=(12, 6))
                plt.subplot(211)
                plt.plot(ci_data.index, ci_data['glucose'], 'b-', label='Glucose')
                plt.axvline(x=event, color='r', linestyle='--', label='Insulin')
                plt.title(f'Glucose and Insulin Data for Event at {event}')
                plt.legend()
                
                plt.subplot(212)
                for col in ci_data.columns:
                    if col not in ['glucose', 'insulin']:
                        plt.plot(ci_data.index, ci_data[col], label=col)
                plt.axvline(x=event, color='r', linestyle='--')
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(f"data_visual/event_{event_idx}_data.png")
                plt.close()
                
                pre_data = ci_data.loc[pre_period[0]:closest_pre].copy()
                for col in pre_data.columns:
                    if pre_data[col].nunique() == 1:
                        constant_val = pre_data[col].iloc[0]
                        ci_data.loc[closest_pre, col] = constant_val + 1
                
                try:
                    impact = CausalImpact(ci_data, pre_period, post_period, prior_level_sd=None)
                    post_inferences = impact.inferences.loc[impact.inferences.index >= post_period[0]]
                    avg_effect = post_inferences['point_effects'].mean()
                    cum_effect = post_inferences['post_cum_effects'].iloc[-1]
                    
                    impact.plot()
                    fig = plt.gcf()
                    fig.savefig(f"data_causal/event_{event_idx}_impact.png")
                    plt.close(fig)
                    
                    event_result = {
                        'event_time': event,
                        'window_start': window_start,
                        'window_end': window_end,
                        'insulin_dose': window_data['insulin'].sum(),
                        'avg_effect': avg_effect,
                        'cum_effect': cum_effect,
                        'pre_points': sum(window_data.index <= closest_pre),
                        'post_points': sum(window_data.index >= closest_post),
                        'pre_period': pre_period,
                        'post_period': post_period
                    }
                    
                    summary_list.append(event_result)
                    model_dict[f"event_{event_idx}"] = {
                        'impact': impact,
                        'ci_data': ci_data,
                        'event_details': event_result
                    }
                    
                    print(f"Success: CausalImpact analysis completed for event at {event}")
                    print(f"Average effect: {avg_effect:.2f}, Cumulative effect: {cum_effect:.2f}")
                    
                except Exception as e:
                    print(f"Error running CausalImpact for event at {event}: {e}")
                    
            except Exception as e:
                print(f"Error processing window data for event at {event}: {e}")
    
    # Save results to pickle file for evaluation script to use
    with open('model_results.pkl', 'wb') as f:
        pickle.dump({
            'filtered_events': filtered_events,
            'summary_list': summary_list,
            'model_dict': model_dict
        }, f)
    
    # Save summary list to CSV for easy review
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv('data_causal/summary_results.csv')
        print(f"\nSaved summary results to data_causal/summary_results.csv")
    
    return filtered_events, summary_list, model_dict

if __name__ == "__main__":
    data_path = "../../synthetic_data/data/ml_dataset.csv"  # Update this path as needed
    run_causal_modeling(data_path)
    print("\nModeling complete. Run evaluation.py to calculate metrics.")