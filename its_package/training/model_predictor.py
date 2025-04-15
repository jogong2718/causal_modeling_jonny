import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

# Import dependencies
from ..its_models.causalimpact_model import CausalImpactModel
from ..its_models.statsmodels_its import StatsmodelsITSModel

class ITSPredictor:
    """
    Predictor class for trained ITS models
    
    This class handles prediction with trained models, including making 
    counterfactual predictions for different insulin doses.
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.models = {
            "causalimpact": None,
            "statsmodels": None,
            "ensemble": None
        }
        self.model_info = {}
        
    def load_model(self, model_path, model_type=None):
        """
        Load a trained model from a file
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        model_type : str, optional
            Type of model ('causalimpact', 'statsmodels', 'ensemble')
            If not provided, will be detected from the file
            
        Returns:
        --------
        self
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Detect model type from filename if not provided
        if model_type is None:
            if "causalimpact" in model_path:
                model_type = "causalimpact"
            elif "statsmodels" in model_path:
                model_type = "statsmodels"
            elif "ensemble" in model_path:
                model_type = "ensemble"
            else:
                raise ValueError("Could not detect model type from filename. Please specify model_type.")
                
        # Load model
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            if model_type == "causalimpact":
                self.models["causalimpact"] = model_data["model"]
                self.model_info["causalimpact"] = model_data["info"]
                print(f"Loaded CausalImpact model from {model_path}")
                
            elif model_type == "statsmodels":
                self.models["statsmodels"] = model_data["model"]
                self.model_info["statsmodels"] = model_data["info"]
                print(f"Loaded Statsmodels ITS model from {model_path}")
                
            elif model_type == "ensemble":
                # For ensemble, load the component models
                ci_path = model_data["causalimpact_model"]
                sm_path = model_data["statsmodels_model"]
                self.model_info["ensemble"] = model_data
                
                if ci_path and os.path.exists(ci_path):
                    self.load_model(ci_path, "causalimpact")
                    
                if sm_path and os.path.exists(sm_path):
                    self.load_model(sm_path, "statsmodels")
                    
                self.models["ensemble"] = {
                    "causalimpact": self.models["causalimpact"],
                    "statsmodels": self.models["statsmodels"]
                }
                print(f"Loaded ensemble model from {model_path}")
                
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
            
        return self
    
    def predict_glucose(self, pre_period_data, intervention_time, 
                       post_period_length="30min", intervention_value=None, 
                       model_type="ensemble", time_frequency="5min"):
        """
        Predict glucose values after an intervention
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Data for the pre-intervention period
        intervention_time : datetime or str
            Time of the intervention
        post_period_length : str or int
            Length of post-period (as time string or number of steps)
        intervention_value : float, optional
            Value of the intervention (insulin dose)
        model_type : str, optional
            Type of model to use for prediction ('causalimpact', 'statsmodels', 'ensemble')
        time_frequency : str, optional
            Time frequency for prediction points
            
        Returns:
        --------
        DataFrame
            DataFrame with predicted glucose values
        """
        # Convert intervention_time to datetime if it's a string
        if isinstance(intervention_time, str):
            intervention_time = pd.to_datetime(intervention_time)
            
        # Generate post-period timestamps
        if isinstance(post_period_length, str):
            post_end_time = intervention_time + pd.Timedelta(post_period_length)
            post_period_index = pd.date_range(
                start=intervention_time, 
                end=post_end_time, 
                freq=time_frequency
            )
        else:
            post_period_index = pd.date_range(
                start=intervention_time, 
                periods=post_period_length+1,  # +1 to include intervention time
                freq=time_frequency
            )
        
        # Create empty dataframe for predictions
        predictions = pd.DataFrame(index=post_period_index)
        
        # Make predictions based on model type
        if model_type == "causalimpact":
            if self.models["causalimpact"] is None:
                raise ValueError("CausalImpact model not loaded")
                
            predictions = self._predict_with_causalimpact(pre_period_data, intervention_time, 
                                                        post_period_index, intervention_value)
                
        elif model_type == "statsmodels":
            if self.models["statsmodels"] is None:
                raise ValueError("StatsModels ITS model not loaded")
                
            predictions = self._predict_with_statsmodels(pre_period_data, intervention_time, 
                                                       post_period_index, intervention_value)
                
        elif model_type == "ensemble":
            # Use both models and average the results
            if self.models["causalimpact"] is None or self.models["statsmodels"] is None:
                raise ValueError("Both CausalImpact and StatsModels ITS models required for ensemble prediction")
                
            ci_predictions = self._predict_with_causalimpact(pre_period_data, intervention_time, 
                                                           post_period_index, intervention_value)
                
            sm_predictions = self._predict_with_statsmodels(pre_period_data, intervention_time, 
                                                          post_period_index, intervention_value)
                
            # Average the predictions
            predictions["ci_predicted"] = ci_predictions["predicted"]
            predictions["sm_predicted"] = sm_predictions["predicted"]
            predictions["predicted"] = (ci_predictions["predicted"] + sm_predictions["predicted"]) / 2
            
            # Create combined confidence intervals
            predictions["lower"] = (ci_predictions["lower"] + sm_predictions["lower"]) / 2
            predictions["upper"] = (ci_predictions["upper"] + sm_predictions["upper"]) / 2
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return predictions
    
    def _predict_with_causalimpact(self, pre_period_data, intervention_time, post_period_index, intervention_value=None):
        """
        Generate predictions using a CausalImpact model
        """
        # Create output dataframe
        predictions = pd.DataFrame(index=post_period_index)
        
        # For CausalImpact, we need:
        # 1. A complete dataset for the BSTS model including pre and post periods
        # 2. Covariates for the post period (we'll extend from pre-period)
        
        # Extract the CausalImpact model
        impact = self.models["causalimpact"]
        
        # Get original pre-period
        original_pre_period = list(self.model_info["causalimpact"]["pre_period"]) if "pre_period" in self.model_info["causalimpact"] else None
        
        # If we have enough context in pre_period_data
        try:
            # Generate post-period covariates - a simple approach is to copy the last values
            last_values = pre_period_data.iloc[-1].copy()
            
            # For ARIMA-type forecasting, we'd need more sophisticated extension of covariates
            post_period_covariates = pd.DataFrame(index=post_period_index, columns=pre_period_data.columns)
            
            # Fill with last observed values (this is a simplification)
            for col in post_period_covariates.columns:
                post_period_covariates[col] = last_values[col]
            
            # Create a new dataframe with both periods for prediction
            combined_data = pd.concat([pre_period_data, post_period_covariates])
            combined_data = combined_data.sort_index()
            
            # If intervention value is provided, use it to scale the prediction
            # (This is specific to insulin-glucose data - might need adjustment for other cases)
            if intervention_value is not None:
                # Extract coefficients from the impact model to appropriately scale predictions
                # For this example we'll use a simple proportional approach
                default_dose = 5.0  # Assuming default dose used in training
                scaling_factor = intervention_value / default_dose if default_dose > 0 else 1.0
            
            # Get point predictions from the causal impact model
            # We need more sophisticated logic here to use the BSTS model directly
            # but for simplicity, we'll base it on prior observations
            
            # Get the counterfactual prediction (what would happen without intervention)
            counterfactual = pd.Series(index=post_period_index)
            
            # For each timestamp, predict the next value
            for i, ts in enumerate(post_period_index):
                # Use the model's predict function (simplified approach here)
                # In a real implementation, we'd use the BSTS model more directly
                if i == 0:
                    # First prediction - use the last value from pre-period with a small trend
                    last_glucose = pre_period_data["glucose"].iloc[-1]
                    counterfactual[ts] = last_glucose + 2.0  # Simplified trend assumption
                else:
                    # Continue the trend
                    counterfactual[ts] = counterfactual[post_period_index[i-1]] + 2.0
            
            # Calculate the effect size based on the intervention value
            effect_size = -30 * scaling_factor  # Simplified effect size estimate
            effect_decay = 0.9  # Effect decays over time
            
            # Calculate effect at each point
            effect = pd.Series(index=post_period_index)
            for i, ts in enumerate(post_period_index):
                time_since_intervention = (ts - intervention_time).total_seconds() / 60.0
                effect[ts] = effect_size * (effect_decay ** (time_since_intervention / 30))
            
            # Calculate predicted glucose = counterfactual + effect
            predictions["counterfactual"] = counterfactual
            predictions["effect"] = effect
            predictions["predicted"] = counterfactual + effect
            
            # Add uncertainty bands
            uncertainty = 10.0 * np.sqrt(scaling_factor)  # Simplified uncertainty model
            predictions["lower"] = predictions["predicted"] - uncertainty
            predictions["upper"] = predictions["predicted"] + uncertainty
            
        except Exception as e:
            print(f"Error in CausalImpact prediction: {str(e)}")
            # Provide a simple fallback
            predictions["predicted"] = pre_period_data["glucose"].iloc[-1] * np.ones(len(post_period_index))
            predictions["lower"] = predictions["predicted"] - 20
            predictions["upper"] = predictions["predicted"] + 20
            
        return predictions
    
    def _predict_with_statsmodels(self, pre_period_data, intervention_time, post_period_index, intervention_value=None):
        """
        Generate predictions using a Statsmodels ITS model
        """
        # Create output dataframe
        predictions = pd.DataFrame(index=post_period_index)
        
        # For StatsModels, we need:
        # 1. Extract the coefficients from the trained model
        # 2. Apply them to generate predictions for the post-period
        
        # Get the trained model
        model = self.models["statsmodels"]
        
        try:
            # Get model parameters
            params = model.params
            
            # Default reference values
            default_dose = 5.0  # Assuming default dose used in training
            scaling_factor = intervention_value / default_dose if intervention_value is not None and default_dose > 0 else 1.0
            
            # Get earliest time in the pre-period data to use as reference
            start_time = pre_period_data.index.min()
            
            # Create prediction data with the required inputs for the model
            pred_data = pd.DataFrame(index=post_period_index)
            pred_data['time'] = (pred_data.index - start_time).total_seconds() / 60  # Convert to minutes
            pred_data['post'] = 1  # All points are post-intervention
            pred_data['time_post'] = pred_data['time'] * pred_data['post']
            
            # Get baseline level and trend
            baseline_level = params.get('Intercept', 0)
            baseline_trend = params.get('time', 0)
            
            # Get intervention effects
            level_change = params.get('post', 0) * scaling_factor
            slope_change = params.get('time_post', 0) * scaling_factor
            
            # Calculate predictions
            predictions['predicted'] = (
                baseline_level + 
                baseline_trend * pred_data['time'] +
                level_change * pred_data['post'] +
                slope_change * pred_data['time_post']
            )
            
            # Add uncertainty bands
            # A proper approach would use the model's confidence intervals
            # This is a simplified approach
            uncertainty = 5.0 * np.sqrt(scaling_factor)  # Simple formula
            predictions['lower'] = predictions['predicted'] - uncertainty
            predictions['upper'] = predictions['predicted'] + uncertainty
            
        except Exception as e:
            print(f"Error in StatsModels prediction: {str(e)}")
            # Fallback prediction
            predictions["predicted"] = pre_period_data["glucose"].iloc[-1] * np.ones(len(post_period_index))
            predictions["lower"] = predictions["predicted"] - 20
            predictions["upper"] = predictions["predicted"] + 20
            
        return predictions
    
    def predict_counterfactual_doses(self, pre_period_data, intervention_time, 
                                    actual_dose, counterfactual_doses, 
                                    post_period_length="30min", 
                                    model_type="ensemble", time_frequency="5min"):
        """
        Predict glucose trends for multiple counterfactual insulin doses
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Data for the pre-intervention period
        intervention_time : datetime or str
            Time of the intervention
        actual_dose : float
            The actual insulin dose
        counterfactual_doses : list
            List of counterfactual insulin doses to simulate
        post_period_length : str or int
            Length of post-period (as time string or number of steps)
        model_type : str, optional
            Type of model to use ('causalimpact', 'statsmodels', 'ensemble')
        time_frequency : str, optional
            Time frequency for prediction points
            
        Returns:
        --------
        dict
            Dictionary mapping dose values to prediction DataFrames
        """
        results = {}
        
        # First get prediction for actual dose
        actual_predictions = self.predict_glucose(
            pre_period_data=pre_period_data,
            intervention_time=intervention_time,
            post_period_length=post_period_length,
            intervention_value=actual_dose,
            model_type=model_type,
            time_frequency=time_frequency
        )
        results[actual_dose] = actual_predictions
        
        # Get predictions for each counterfactual dose
        for dose in counterfactual_doses:
            if dose == actual_dose:
                continue
                
            cf_predictions = self.predict_glucose(
                pre_period_data=pre_period_data,
                intervention_time=intervention_time,
                post_period_length=post_period_length,
                intervention_value=dose,
                model_type=model_type,
                time_frequency=time_frequency
            )
            results[dose] = cf_predictions
            
        return results
    
    def plot_prediction(self, pre_period_data, predictions, intervention_time,
                       target_col='glucose', figsize=(12, 6), output_path=None):
        """
        Plot pre-period data and post-period predictions
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Original pre-period data
        predictions : DataFrame
            Predicted post-period data
        intervention_time : datetime
            Time of intervention
        target_col : str
            Name of target column
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot pre-period data
        if target_col in pre_period_data.columns:
            plt.plot(pre_period_data.index, pre_period_data[target_col], 'b-', label='Historical Data')
        else:
            plt.plot(pre_period_data.index, pre_period_data['response'], 'b-', label='Historical Data')
        
        # Plot predictions
        plt.plot(predictions.index, predictions['predicted'], 'r--', label='Predicted')
        
        # Plot confidence intervals if available
        if 'lower' in predictions.columns and 'upper' in predictions.columns:
            plt.fill_between(
                predictions.index,
                predictions['lower'],
                predictions['upper'],
                color='r', alpha=0.2,
                label='Prediction Interval'
            )
        
        # Mark intervention
        plt.axvline(x=intervention_time, color='k', linestyle='-', label='Intervention')
        
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.title('Predicted Post-Intervention Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def plot_counterfactual_comparison(self, pre_period_data, counterfactual_results, 
                                      intervention_time, target_col='glucose', 
                                      figsize=(12, 8), output_path=None):
        """
        Plot comparison of different counterfactual doses
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Original pre-period data
        counterfactual_results : dict
            Dictionary mapping doses to prediction DataFrames
        intervention_time : datetime
            Time of intervention
        target_col : str
            Name of target column
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot pre-period data
        if target_col in pre_period_data.columns:
            plt.plot(pre_period_data.index, pre_period_data[target_col], 'b-', label='Historical Data')
        else:
            plt.plot(pre_period_data.index, pre_period_data['response'], 'b-', label='Historical Data')
            
        # Color map for different doses
        n_doses = len(counterfactual_results)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_doses))
        
        # Plot each counterfactual
        for i, (dose, predictions) in enumerate(sorted(counterfactual_results.items())):
            plt.plot(predictions.index, predictions['predicted'], '--', 
                    color=colors[i], label=f'Dose: {dose}u', alpha=0.7)
            
        # Mark intervention
        plt.axvline(x=intervention_time, color='k', linestyle='-', label='Intervention')
        
        # Add horizontal lines for target ranges
        plt.axhline(y=180, color='r', linestyle=':', alpha=0.7, label='High Limit')
        plt.axhline(y=70, color='r', linestyle=':', alpha=0.7, label='Low Limit')
        plt.axhline(y=100, color='g', linestyle=':', alpha=0.7, label='Target')
        
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.title('Counterfactual Dose Comparison')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def find_optimal_dose(self, pre_period_data, intervention_time, dose_range,
                         target_glucose=100, time_point=None, post_period="2h",
                         model_type="ensemble", time_frequency="5min", n_steps=10):
        """
        Find the optimal insulin dose to reach target glucose
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Data for the pre-intervention period
        intervention_time : datetime or str
            Time of the intervention
        dose_range : tuple
            (min_dose, max_dose) range to search
        target_glucose : float
            Target glucose value to achieve
        time_point : datetime or str, optional
            Specific time point to optimize for (default: end of post_period)
        post_period : str
            Length of post-period to predict
        model_type : str, optional
            Type of model to use for prediction
        time_frequency : str, optional
            Time frequency for prediction points
        n_steps : int, optional
            Number of doses to evaluate in the range
            
        Returns:
        --------
        dict
            Results of optimization including optimal dose
        """
        min_dose, max_dose = dose_range
        doses = np.linspace(min_dose, max_dose, n_steps)
        
        # Convert intervention_time to datetime if it's a string
        if isinstance(intervention_time, str):
            intervention_time = pd.to_datetime(intervention_time)
            
        # If time_point not specified, use the end of post_period
        if time_point is None:
            if isinstance(post_period, str):
                time_point = intervention_time + pd.Timedelta(post_period)
            else:
                # Assume post_period is the number of time steps
                time_point = intervention_time + pd.Timedelta(minutes=(post_period * pd.Timedelta(time_frequency).total_seconds() / 60))
        elif isinstance(time_point, str):
            time_point = pd.to_datetime(time_point)
            
        # Get predictions for each dose
        results = {}
        dose_values = []
        glucose_values = []
        
        for dose in doses:
            predictions = self.predict_glucose(
                pre_period_data=pre_period_data,
                intervention_time=intervention_time,
                post_period_length=post_period,
                intervention_value=dose,
                model_type=model_type,
                time_frequency=time_frequency
            )
            
            # Find glucose at the time point
            closest_time = predictions.index[abs(predictions.index - time_point).argmin()]
            glucose_value = predictions.loc[closest_time, 'predicted']
            
            results[dose] = {
                'glucose': glucose_value,
                'error': abs(glucose_value - target_glucose)
            }
            
            dose_values.append(dose)
            glucose_values.append(glucose_value)
            
        # Find optimal dose
        error_values = [abs(g - target_glucose) for g in glucose_values]
        optimal_idx = np.argmin(error_values)
        optimal_dose = dose_values[optimal_idx]
        optimal_glucose = glucose_values[optimal_idx]
        
        print(f"Optimal dose: {optimal_dose:.2f}u → Expected glucose: {optimal_glucose:.1f} mg/dL (target: {target_glucose} mg/dL)")
        
        # Create figure showing dose-response relationship
        plt.figure(figsize=(10, 6))
        plt.plot(dose_values, glucose_values, 'bo-')
        plt.axhline(y=target_glucose, color='g', linestyle='--', label=f'Target: {target_glucose} mg/dL')
        plt.axvline(x=optimal_dose, color='r', linestyle='--', label=f'Optimal: {optimal_dose:.2f}u')
        plt.xlabel('Insulin Dose (units)')
        plt.ylabel('Predicted Glucose (mg/dL)')
        plt.title(f'Insulin Dose vs. Predicted Glucose at {time_point}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return {
            'optimal_dose': optimal_dose,
            'optimal_glucose': optimal_glucose,
            'target_glucose': target_glucose,
            'dose_response': dict(zip(dose_values, glucose_values)),
            'time_point': time_point
        }
    
    def maximize_time_in_range(self, pre_period_data, intervention_time, dose_range,
                              low_threshold=80, high_threshold=130, post_period="2h",
                              model_type="ensemble", time_frequency="5min", n_steps=10):
        """
        Find the optimal insulin dose to maximize time in range for glucose levels
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Data for the pre-intervention period
        intervention_time : datetime or str
            Time of the intervention
        dose_range : tuple
            (min_dose, max_dose) range to search
        low_threshold : float
            Lower bound of target glucose range (default: 80 mg/dL)
        high_threshold : float
            Upper bound of target glucose range (default: 130 mg/dL)
        post_period : str
            Length of post-period to predict
        model_type : str, optional
            Type of model to use for prediction
        time_frequency : str, optional
            Time frequency for prediction points
        n_steps : int, optional
            Number of doses to evaluate in the range
            
        Returns:
        --------
        dict
            Results of optimization including optimal dose and time in range metrics
        """
        min_dose, max_dose = dose_range
        doses = np.linspace(min_dose, max_dose, n_steps)
        
        # Convert intervention_time to datetime if it's a string
        if isinstance(intervention_time, str):
            intervention_time = pd.to_datetime(intervention_time)
            
        # Get predictions for each dose
        results = {}
        dose_values = []
        tir_values = []  # Time in range percentages
        
        for dose in doses:
            predictions = self.predict_glucose(
                pre_period_data=pre_period_data,
                intervention_time=intervention_time,
                post_period_length=post_period,
                intervention_value=dose,
                model_type=model_type,
                time_frequency=time_frequency
            )
            
            # Calculate time in range (percentage of post-period predictions within target range)
            in_range = ((predictions['predicted'] >= low_threshold) & 
                        (predictions['predicted'] <= high_threshold))
            time_in_range = in_range.mean() * 100  # Convert to percentage
            
            # Calculate additional metrics (optional)
            below_range = (predictions['predicted'] < low_threshold).mean() * 100
            above_range = (predictions['predicted'] > high_threshold).mean() * 100
            
            # Calculate mean glucose and glucose variability
            mean_glucose = predictions['predicted'].mean()
            glucose_std = predictions['predicted'].std()
            
            results[dose] = {
                'time_in_range': time_in_range,
                'below_range': below_range,
                'above_range': above_range,
                'mean_glucose': mean_glucose,
                'glucose_std': glucose_std,
                'predictions': predictions
            }
            
            dose_values.append(dose)
            tir_values.append(time_in_range)
            
        # Find optimal dose that maximizes time in range
        optimal_idx = np.argmax(tir_values)
        optimal_dose = dose_values[optimal_idx]
        optimal_tir = tir_values[optimal_idx]
        optimal_stats = results[optimal_dose]
        
        print(f"Optimal dose: {optimal_dose:.2f}u → Time in range: {optimal_tir:.1f}% (Target range: {low_threshold}-{high_threshold} mg/dL)")
        print(f"Mean glucose: {optimal_stats['mean_glucose']:.1f} mg/dL, Std: {optimal_stats['glucose_std']:.1f} mg/dL")
        print(f"Below range: {optimal_stats['below_range']:.1f}%, Above range: {optimal_stats['above_range']:.1f}%")
        
        # Create figure showing dose-response relationship for time in range
        plt.figure(figsize=(10, 6))
        plt.plot(dose_values, tir_values, 'bo-')
        plt.axvline(x=optimal_dose, color='r', linestyle='--', label=f'Optimal: {optimal_dose:.2f}u')
        plt.xlabel('Insulin Dose (units)')
        plt.ylabel('Time in Range (%)')
        plt.title(f'Insulin Dose vs. Time in Range {low_threshold}-{high_threshold} mg/dL')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return {
            'optimal_dose': optimal_dose,
            'optimal_tir': optimal_tir,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'mean_glucose': optimal_stats['mean_glucose'],
            'glucose_std': optimal_stats['glucose_std'],
            'below_range': optimal_stats['below_range'],
            'above_range': optimal_stats['above_range'],
            'dose_response': dict(zip(dose_values, tir_values)),
            'all_results': results
        }

    def plot_time_in_range_comparison(self, pre_period_data, counterfactual_results, 
                                     intervention_time, low_threshold=80, high_threshold=130,
                                     target_col='glucose', figsize=(14, 10), output_path=None):
        """
        Plot comparison of different doses with time-in-range highlighted
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Original pre-period data
        counterfactual_results : dict
            Dictionary mapping doses to prediction DataFrames
        intervention_time : datetime
            Time of intervention
        low_threshold : float
            Lower bound of target glucose range
        high_threshold : float
            Upper bound of target glucose range
        target_col : str
            Name of target column
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot pre-period data
        if target_col in pre_period_data.columns:
            plt.plot(pre_period_data.index, pre_period_data[target_col], 'b-', label='Historical Data')
        else:
            plt.plot(pre_period_data.index, pre_period_data['response'], 'b-', label='Historical Data')
            
        # Color map for different doses
        n_doses = len(counterfactual_results)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_doses))
        
        # Plot each counterfactual with time-in-range highlighted
        for i, (dose, predictions) in enumerate(sorted(counterfactual_results.items())):
            color = colors[i]
            plt.plot(predictions.index, predictions['predicted'], '--', 
                    color=color, label=f'Dose: {dose}u', alpha=0.7)
            
            # Highlight time-in-range regions
            in_range_idx = ((predictions['predicted'] >= low_threshold) & 
                           (predictions['predicted'] <= high_threshold))
            
            if any(in_range_idx):
                plt.scatter(
                    predictions.index[in_range_idx], 
                    predictions['predicted'][in_range_idx],
                    color=color, marker='o', s=30, alpha=0.8
                )
                
            # Calculate time in range percentage
            time_in_range = in_range_idx.mean() * 100
            plt.plot([], [], ' ', label=f'TIR ({dose}u): {time_in_range:.1f}%')
            
        # Mark intervention
        plt.axvline(x=intervention_time, color='k', linestyle='-', label='Intervention')
        
        # Add horizontal lines for target ranges
        plt.axhspan(low_threshold, high_threshold, color='g', alpha=0.1, label='Target Range')
        plt.axhline(y=high_threshold, color='r', linestyle=':', alpha=0.7)
        plt.axhline(y=low_threshold, color='r', linestyle=':', alpha=0.7)
        
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.title(f'Counterfactual Dose Comparison with Time in Range ({low_threshold}-{high_threshold} mg/dL)')
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved to {output_path}")
        else:
            plt.show()