import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from causalimpact import CausalImpact
from .base import BaseITSModel
from datetime import datetime

# Suppress specific pandas warnings from CausalImpact
warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated", category=FutureWarning)

class CausalImpactModel(BaseITSModel):
    """ITS model using CausalImpact."""
    
    def __init__(self, output_dir=None):
        """Initialize the CausalImpact model."""
        super().__init__(name="CausalImpactModel", output_dir=output_dir)
        self.impact = None
        self.data = None
        self.pre_period = None
        self.post_period = None
        
    def fit(self, data, pre_period, post_period):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to fit the model to
        pre_period : list
            [start, end] of pre-intervention period
        post_period : list
            [start, end] of post-intervention period
        prior_level_sd : float
            Standard deviation of the prior level
            
        Returns:
        --------
        self
        """
        # Handle edge cases first
        if data.empty:
            raise ValueError("Empty data provided for model fitting")
        
        # Ensure we have at least one predictor variable
        if data.shape[1] <= 1:
            raise ValueError("Data must have at least one predictor variable")
        
        # Clean data - replace infinities, drop NAs, and ensure column variability
        clean_data = data.replace([np.inf, -np.inf], np.nan)
        clean_data = clean_data.dropna()
        
        # Keep only columns with variation (more than one unique value)
        clean_data = clean_data.loc[:, clean_data.nunique() > 1]
        
        # Check if we still have enough data
        if clean_data.empty or clean_data.shape[1] <= 1:
            raise ValueError("After cleaning, data is insufficient for modeling")
        
        # Handle columns with constant values in pre-period
        pre_data = clean_data.loc[pre_period[0]:pre_period[1]].copy()
        for col in pre_data.columns:
            if pre_data[col].nunique() == 1:
                # Add small noise to avoid constant values
                std_dev = clean_data[col].std()
                if std_dev == 0:
                    std_dev = 0.01 * clean_data[col].mean()
                    if std_dev == 0:
                        std_dev = 0.1  # Fallback if mean is also 0
                
                # Add small random noise to the column
                clean_data[col] = clean_data[col] + np.random.normal(0, std_dev/10, len(clean_data))
        
        # Store processed data
        self.data = clean_data
        self.pre_period = pre_period
        self.post_period = post_period
        
        # Fit the model
        self.impact = CausalImpact(clean_data, pre_period, post_period)
        self.impact.run()
        
        return self
    
    def get_results(self):
        """
        Get the results of the model.
        
        Returns:
        --------
        dict
            Dictionary of results
        """
        if self.impact is None:
            raise ValueError("Model has not been fit yet")
        
        # Extract key results
        post_inferences = self.impact.inferences.loc[self.impact.inferences.index >= self.post_period[0]]
        
        # Calculate mean effect
        avg_effect = post_inferences['point_effect'].mean()
        cum_effect = post_inferences['cum_effect'].iloc[-1]
        
        # Calculate p-value
        p_value = self.impact.summary_data.get('p', 1.0)
        
        # Calculate confidence intervals
        effect_lower = post_inferences['point_effects_lower'].mean()
        effect_upper = post_inferences['point_effects_upper'].mean()
        
        results = {
            'avg_effect': avg_effect,
            'cum_effect': cum_effect,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_lower': effect_lower,
            'effect_upper': effect_upper,
            'post_data': post_inferences
        }
        
        return results
    
    def plot(self, fig=None, figsize=(12, 8)):
        """
        Plot the results of the model.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to plot on
        figsize : tuple
            Size of the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the plot
        """
        if self.impact is None:
            raise ValueError("Model has not been fit yet")
        
        if fig is None:
            fig = plt.figure(figsize=figsize)
        
        self.impact.plot(panels=['original', 'pointwise', 'cumulative'])
        plt.tight_layout()
        
        return fig
    
    def plot_original_data(self, target_col='glucose', figsize=(12, 6)):
        """
        Plot the original data with intervention point.
        
        Parameters:
        -----------
        target_col : str
            Name of the target column
        figsize : tuple
            Size of the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the plot
        """
        if self.data is None:
            raise ValueError("No data available")
        
        fig = plt.figure(figsize=figsize)
        
        # Plot the data
        plt.plot(self.data.index, self.data[target_col], 'b-', label=target_col)
        
        # Mark the intervention
        intervention_time = self.pre_period[1]
        plt.axvline(x=intervention_time, color='r', linestyle='--', label='Intervention')
        
        # Add labels and legend
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.title('Time Series Data with Intervention Point')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return fig
    
    def save_plot(self, filename=None):
        """
        Save the plot to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the plot to
        """
        if self.output_dir is None:
            raise ValueError("Output directory not set")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"causalimpact_plot_{timestamp}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create the plot
        fig = self.plot()
        
        # Save the plot
        fig.savefig(filepath)
        plt.close(fig)
        
        print(f"Plot saved to {filepath}")
        
    def save_model(self, filename=None):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the model to
        """
        if self.output_dir is None:
            self.output_dir = os.path.dirname(filename)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"causalimpact_model_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
        else:
            filepath = filename
        
        if self.impact is None:
            raise ValueError("Model has not been fit yet")
        
        # Extract summary data from the impact object
        # The CausalImpact object might not have summary_data directly
        # Instead we'll use the data we can access safely
        try:
            # Try to get summary from report if available
            report = self.impact.summary()
            summary_data = {'report': report}
        except:
            # Fallback to a more basic summary
            summary_data = {}
        
        # Add model parameters and results data that's safely accessible
        post_data = self.impact.inferences.loc[self.impact.inferences.index >= self.post_period[0]]
        avg_effect = float(post_data['point_effect'].mean())
        cum_effect = float(post_data['cum_effect'].iloc[-1])
        
        # Convert to serializable format
        results = {
            'pre_period': [str(x) for x in self.pre_period],
            'post_period': [str(x) for x in self.post_period],
            'avg_effect': avg_effect,
            'cum_effect': cum_effect,
            'summary': summary_data
        }
        
        # Save to file - use pickle instead of json for more complete object serialization
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        # Also save the entire model object for full preservation
        model_filepath = filepath.replace('.json', '.pkl') if filepath.endswith('.json') else filepath
        with open(model_filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {model_filepath}")
        
        return model_filepath
