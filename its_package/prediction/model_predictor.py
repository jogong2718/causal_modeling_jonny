import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class InterventionPredictor:
    """
    Class for making predictions with pre-trained causal models.
    This allows using models in production to forecast post-intervention values.
    """

    def __init__(self, model_path=None):
        """
        Initialize the predictor with an optional pre-trained model.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to saved model file (.pkl)
        """
        self.model = None
        self.model_type = None
        self.prediction_horizon = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a saved model from file.
        
        Parameters:
        -----------
        model_path : str
            Path to saved model file (.pkl)
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['type']
        self.model_info = model_data.get('info', {})
        
        print(f"Loaded {self.model_type} model from {model_path}")
        return self
    
    def predict(self, pre_period_data, intervention_time, post_period_length, 
                intervention_value=None, time_frequency='5min'):
        """
        Predict post-intervention values based on pre-period data.
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Time series data for the pre-intervention period
        intervention_time : datetime or str
            Time of intervention
        post_period_length : str or int
            Length of post period (e.g., '1h', '2d' or number of points)
        intervention_value : float, optional
            Value of intervention (for models that use this)
        time_frequency : str, optional
            Frequency of time series for generated timestamps
            
        Returns:
        --------
        DataFrame
            Predicted post-intervention data
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Convert intervention_time to datetime if it's a string
        if isinstance(intervention_time, str):
            intervention_time = pd.to_datetime(intervention_time)
            
        # Generate post-period timepoints
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
        
        # Call the appropriate prediction method based on model type
        if self.model_type == 'causalimpact':
            return self._predict_with_causalimpact(pre_period_data, intervention_time, predictions)
        
        elif self.model_type == 'statsmodels_its':
            return self._predict_with_statsmodels(pre_period_data, intervention_time, predictions, intervention_value)
        
        else:
            raise ValueError(f"Prediction with model type '{self.model_type}' not implemented")
    
    def _predict_with_causalimpact(self, pre_period_data, intervention_time, predictions):
        """
        Generate predictions using a CausalImpact model.
        
        For CausalImpact, we:
        1. Use the learned relationships between response & covariates
        2. Project those into the post-period using post-period covariates
        3. Return counterfactual predictions
        """
        # Extract model components
        bsts_model = self.model.model
        
        # We need covariates for the post period
        # Here we assume they're provided or we could forecast them
        # For simplicity, we'll just use the last values from pre period
        # In production, you'd want to provide real covariate values or forecasts
        
        # Create post-period covariates (simplified approach)
        post_period_covariates = pd.DataFrame(
            index=predictions.index,
            columns=pre_period_data.columns.drop('response')  # All columns except response
        )
        
        # Fill with last values from pre-period (very simple approach)
        for col in post_period_covariates.columns:
            post_period_covariates[col] = pre_period_data[col].iloc[-1]
        
        # Combine pre-period and post-period 
        combined_data = pd.concat([
            pre_period_data,
            pd.DataFrame(index=predictions.index, columns=pre_period_data.columns)
        ]).sort_index()
        
        # Fill post-period covariates
        for col in combined_data.columns:
            if col != 'response':
                combined_data.loc[predictions.index, col] = post_period_covariates[col]
                
        # Use the original model to predict
        # This is a simplified approach - in a real implementation, you would
        # properly use the BSTS model's predict function
        pred_result = self.model.model.predict(
            combined_data, 
            combined_data.index.min(), 
            intervention_time
        )
        
        # Extract predictions for post period
        predictions['predicted'] = pred_result.iloc[-len(predictions):]
        predictions['lower'] = predictions['predicted'] * 0.9  # Simplified
        predictions['upper'] = predictions['predicted'] * 1.1  # Simplified
        
        return predictions
        
    def _predict_with_statsmodels(self, pre_period_data, intervention_time, predictions, intervention_value=None):
        """
        Generate predictions using a Statsmodels ITS model.
        
        This uses the fitted regression model to predict post-intervention values.
        """
        # For statsmodels, we need to:
        # 1. Create a dataframe with post-period time values
        # 2. Set the intervention indicator
        # 3. Use the model to predict
        
        # Prepare prediction data
        pred_data = pd.DataFrame(index=predictions.index)
        
        # Add time variable (minutes since start)
        start_time = pre_period_data.index.min()
        pred_data['time'] = (pred_data.index - start_time).total_seconds() / 60
        
        # Add post-intervention indicator
        pred_data['post'] = 1  # All points are post-intervention
        
        # Add interaction term
        pred_data['time_post'] = pred_data['time'] * pred_data['post']
        
        # Make predictions
        predictions['predicted'] = self.model.predict(pred_data)
        
        # Add confidence intervals (simplified)
        predictions['lower'] = predictions['predicted'] - 10
        predictions['upper'] = predictions['predicted'] + 10
        
        return predictions
    
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
    
    def predict_dose_counterfactuals(self, pre_period_data, intervention_time, post_period_length, 
                                    actual_dose, counterfactual_doses, time_frequency='5min'):
        """
        Predict glucose trends under different insulin doses.
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Time series data for the pre-intervention period
        intervention_time : datetime or str
            Time of intervention
        post_period_length : str or int
            Length of post period (e.g., '1h', '2d' or number of points)
        actual_dose : float
            The actual insulin dose used in the training data
        counterfactual_doses : list of float
            List of alternative doses to simulate
        time_frequency : str, optional
            Frequency of time series for generated timestamps
            
        Returns:
        --------
        dict
            Dictionary of predicted glucose trends for each dose
        """
        results = {}
        
        # First get the baseline prediction with actual dose
        baseline_prediction = self.predict(
            pre_period_data, intervention_time, post_period_length, 
            intervention_value=actual_dose, time_frequency=time_frequency
        )
        results['actual'] = baseline_prediction
        
        # Process each counterfactual dose
        for dose in counterfactual_doses:
            if self.model_type == 'causalimpact':
                # For CausalImpact, scale the effect based on dose ratio
                cf_prediction = self._predict_causalimpact_dose_counterfactual(
                    pre_period_data, intervention_time, post_period_length, 
                    actual_dose, dose, time_frequency
                )
                
            elif self.model_type == 'statsmodels_its':
                # For StatsModels, adjust coefficients based on dose ratio
                cf_prediction = self._predict_statsmodels_dose_counterfactual(
                    pre_period_data, intervention_time, post_period_length, 
                    actual_dose, dose, time_frequency
                )
                
            else:
                raise ValueError(f"Counterfactual dose prediction not implemented for model type '{self.model_type}'")
            
            results[f'dose_{dose}'] = cf_prediction
            
        return results
    
    def _predict_causalimpact_dose_counterfactual(self, pre_period_data, intervention_time, 
                                                post_period_length, actual_dose, new_dose, 
                                                time_frequency):
        """
        Predict CausalImpact counterfactual for different insulin dose.
        
        This uses a dose-response scaling approach based on the assumption that
        insulin effects are approximately proportional to dose.
        """
        # Get the counterfactual prediction (what would happen without insulin)
        no_intervention = self._get_causalimpact_counterfactual(
            pre_period_data, intervention_time, post_period_length, time_frequency
        )
        
        # Get the actual intervention prediction
        actual_prediction = self.predict(
            pre_period_data, intervention_time, post_period_length, 
            intervention_value=actual_dose, time_frequency=time_frequency
        )
        
        # Calculate the effect size from the actual dose
        effect = actual_prediction['predicted'] - no_intervention['predicted']
        
        # Scale the effect by the dose ratio
        dose_ratio = new_dose / actual_dose if actual_dose != 0 else 0
        scaled_effect = effect * dose_ratio
        
        # Create new prediction by adjusting the counterfactual
        new_prediction = no_intervention.copy()
        new_prediction['predicted'] = no_intervention['predicted'] + scaled_effect
        
        # Adjust confidence intervals
        effect_uncertainty_factor = np.sqrt(dose_ratio) # Uncertainty grows with dose
        new_prediction['lower'] = new_prediction['predicted'] - (actual_prediction['predicted'] - actual_prediction['lower']) * effect_uncertainty_factor
        new_prediction['upper'] = new_prediction['predicted'] + (actual_prediction['upper'] - actual_prediction['predicted']) * effect_uncertainty_factor
        
        return new_prediction
    
    def _get_causalimpact_counterfactual(self, pre_period_data, intervention_time, 
                                       post_period_length, time_frequency):
        """Get the CausalImpact counterfactual prediction (no intervention)"""
        # Generate post-period timepoints
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
        
        # Create dataframe for predictions
        predictions = pd.DataFrame(index=post_period_index)
        
        # Use the model's counterfactual predictions directly
        # This is what CausalImpact calculates as "point_pred"
        impact = self.model
        
        # The model predicts what would have happened without intervention
        point_preds = impact.inferences['point_pred']
        pred_lower = impact.inferences['point_pred_lower']
        pred_upper = impact.inferences['point_pred_upper']
        
        # Match the post-period timestamps - might need interpolation
        # Here using a simple approach - could be enhanced with proper interpolation
        
        # For simplicity, just extract the prediction part that matches our post period
        predictions['predicted'] = point_preds.reindex(post_period_index, method='nearest')
        predictions['lower'] = pred_lower.reindex(post_period_index, method='nearest')
        predictions['upper'] = pred_upper.reindex(post_period_index, method='nearest')
        
        return predictions
    
    def _predict_statsmodels_dose_counterfactual(self, pre_period_data, intervention_time, 
                                               post_period_length, actual_dose, new_dose, 
                                               time_frequency):
        """
        Predict StatsModels counterfactual for different insulin dose.
        
        This adjusts the level_change and slope_change coefficients proportionally
        to the dose change.
        """
        # Create prediction dataframe
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
                periods=post_period_length+1,
                freq=time_frequency
            )
            
        predictions = pd.DataFrame(index=post_period_index)
        
        # Calculate time variable
        start_time = pre_period_data.index.min()
        predictions['time'] = (predictions.index - start_time).total_seconds() / 60
        
        # Get the original model parameters
        params = self.model.params.copy()
        
        # Scale intervention effects by dose ratio
        dose_ratio = new_dose / actual_dose if actual_dose != 0 else 0
        
        # Create prediction data with scaled coefficients
        pred_data = pd.DataFrame(index=post_period_index)
        pred_data['time'] = (pred_data.index - start_time).total_seconds() / 60
        pred_data['post'] = 1  # All points are post-intervention
        pred_data['time_post'] = pred_data['time'] * pred_data['post']
        
        # Original prediction formula: intercept + time*β₁ + post*β₂ + time_post*β₃
        # We scale β₂ (level change) and β₃ (slope change) by dose ratio
        
        predictions['predicted'] = (params['Intercept'] + 
                                   params['time'] * pred_data['time'] + 
                                   params['post'] * dose_ratio * pred_data['post'] + 
                                   params['time_post'] * dose_ratio * pred_data['time_post'])
        
        # Simple confidence intervals (can be refined with proper error propagation)
        uncertainty = np.sqrt(dose_ratio) * 5  # Simplified approach
        predictions['lower'] = predictions['predicted'] - uncertainty
        predictions['upper'] = predictions['predicted'] + uncertainty
        
        return predictions
        
    def plot_dose_counterfactuals(self, pre_period_data, counterfactual_results, intervention_time, 
                                target_col='glucose', figsize=(12, 8), output_path=None):
        """
        Plot pre-period data and multiple dose counterfactuals
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Original pre-period data
        counterfactual_results : dict
            Results from predict_dose_counterfactuals
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
        
        # Define a color map for different doses
        colors = ['r', 'g', 'purple', 'orange', 'c', 'm', 'y', 'k']
        color_idx = 0
        
        # Plot each counterfactual prediction
        for label, prediction_df in counterfactual_results.items():
            color = colors[color_idx % len(colors)]
            if label == 'actual':
                linestyle = '-'
                label_text = 'Actual Dose'
                alpha = 0.9
                linewidth = 2
            else:
                linestyle = '--'
                dose = label.split('_')[1]
                label_text = f'Dose {dose}'
                alpha = 0.7
                linewidth = 1.5
                
            plt.plot(prediction_df.index, prediction_df['predicted'], 
                    color=color, linestyle=linestyle, label=label_text,
                    alpha=alpha, linewidth=linewidth)
            
            # Only show confidence intervals for actual prediction to avoid clutter
            if label == 'actual' and 'lower' in prediction_df.columns and 'upper' in prediction_df.columns:
                plt.fill_between(
                    prediction_df.index,
                    prediction_df['lower'],
                    prediction_df['upper'],
                    color=color, alpha=0.2
                )
                
            color_idx += 1
        
        # Mark intervention
        plt.axvline(x=intervention_time, color='k', linestyle='-', label='Intervention')
        
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.title('Predicted Glucose Response to Different Insulin Doses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def find_optimal_dose(self, pre_period_data, intervention_time, dose_range,
                       target_glucose=100, time_point=None, model_type="ensemble", n_steps=10):
        """
        Find the optimal insulin dose to achieve a target glucose level at a specific time.
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Data for the pre-intervention period
        intervention_time : datetime or str
            Time of intervention
        dose_range : tuple
            (min_dose, max_dose) range to search
        target_glucose : float
            Target glucose level to aim for
        time_point : datetime or str
            Time point at which to achieve target glucose
        model_type : str, optional
            Type of model to use for prediction
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
            
        # Set time_point if not provided
        if time_point is None:
            time_point = intervention_time + pd.Timedelta('2h')  # Default 2 hours after intervention
        elif isinstance(time_point, str):
            if "+" in time_point:  # Relative time format like "+2h"
                offset = time_point.strip("+")
                time_point = intervention_time + pd.Timedelta(offset)
            else:
                time_point = pd.to_datetime(time_point)
        
        # Get predictions for each dose
        predictions_at_time = {}
        dose_values = []
        glucose_values = []
        
        for dose in doses:
            prediction = self.predict(
                pre_period_data=pre_period_data,
                intervention_time=intervention_time,
                post_period_length=time_point - intervention_time,
                intervention_value=dose,
                model_type=model_type
            )
            
            # Get predicted glucose at time_point
            if time_point in prediction.index:
                glucose_at_time = prediction.loc[time_point, 'predicted']
            else:
                # Find closest time point
                closest_time = prediction.index[prediction.index <= time_point].max()
                if pd.isna(closest_time):
                    closest_time = prediction.index[prediction.index >= time_point].min()
                glucose_at_time = prediction.loc[closest_time, 'predicted']
            
            predictions_at_time[dose] = {
                'glucose': glucose_at_time,
                'prediction': prediction
            }
            
            dose_values.append(dose)
            glucose_values.append(glucose_at_time)
        
        # Find optimal dose (closest to target)
        differences = [abs(g - target_glucose) for g in glucose_values]
        optimal_idx = np.argmin(differences)
        optimal_dose = dose_values[optimal_idx]
        optimal_glucose = glucose_values[optimal_idx]
        
        print(f"Optimal dose: {optimal_dose:.2f}u → Predicted glucose at {time_point}: {optimal_glucose:.1f} mg/dL (Target: {target_glucose} mg/dL)")
        
        # Create figure showing dose-response relationship
        plt.figure(figsize=(10, 6))
        plt.plot(dose_values, glucose_values, 'bo-')
        plt.axvline(x=optimal_dose, color='r', linestyle='--', label=f'Optimal: {optimal_dose:.2f}u')
        plt.axhline(y=target_glucose, color='g', linestyle='--', label=f'Target: {target_glucose} mg/dL')
        plt.xlabel('Insulin Dose (units)')
        plt.ylabel('Predicted Glucose (mg/dL)')
        plt.title(f'Insulin Dose vs. Predicted Glucose at {time_point}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return {
            'optimal_dose': optimal_dose,
            'optimal_glucose': optimal_glucose,
            'target_glucose': target_glucose,
            'time_point': time_point,
            'dose_response': dict(zip(dose_values, glucose_values)),
            'all_predictions': predictions_at_time
        }
        
    def predict_glucose(self, pre_period_data, intervention_time, post_period_length, 
                      intervention_value=None, model_type="causalimpact", time_frequency='5min'):
        """
        Predict glucose values after an intervention.
        This is a wrapper around predict() that supports ensemble predictions.
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Time series data for the pre-intervention period
        intervention_time : datetime or str
            Time of intervention
        post_period_length : str
            Length of post period (e.g., '1h', '2h')
        intervention_value : float, optional
            Insulin dose or other intervention value
        model_type : str
            Type of model to use: 'causalimpact', 'statsmodels', or 'ensemble'
        time_frequency : str
            Frequency of time series data
            
        Returns:
        --------
        DataFrame
            Predicted glucose values
        """
        # Convert intervention_time to datetime if it's a string
        if isinstance(intervention_time, str):
            intervention_time = pd.to_datetime(intervention_time)
            
        # For ensemble prediction, we need both model types
        if model_type == "ensemble":
            # Get predictions from both models
            self.model_type = "causalimpact"
            ci_predictions = self.predict(
                pre_period_data, 
                intervention_time, 
                post_period_length, 
                intervention_value,
                time_frequency
            )
            
            self.model_type = "statsmodels_its"
            sm_predictions = self.predict(
                pre_period_data, 
                intervention_time, 
                post_period_length, 
                intervention_value,
                time_frequency
            )
            
            # Combine predictions (simple average)
            combined_predictions = ci_predictions.copy()
            combined_predictions['predicted'] = (ci_predictions['predicted'] + sm_predictions['predicted']) / 2
            
            # Average confidence intervals too
            if 'lower' in ci_predictions.columns and 'lower' in sm_predictions.columns:
                combined_predictions['lower'] = (ci_predictions['lower'] + sm_predictions['lower']) / 2
                combined_predictions['upper'] = (ci_predictions['upper'] + sm_predictions['upper']) / 2
                
            return combined_predictions
            
        else:
            # Use the specified model type
            self.model_type = model_type
            return self.predict(
                pre_period_data, 
                intervention_time, 
                post_period_length, 
                intervention_value,
                time_frequency
            )
    
    def maximize_time_in_range(self, pre_period_data, intervention_time, dose_range,
                              low_threshold=80, high_threshold=130, post_period="2h", 
                              model_type="ensemble", n_steps=10):
        """
        Find the optimal insulin dose that maximizes time in range (TIR) for glucose.
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Data for the pre-intervention period
        intervention_time : datetime or str
            Time of intervention
        dose_range : tuple
            (min_dose, max_dose) range to search
        low_threshold : float
            Lower bound of target glucose range
        high_threshold : float
            Upper bound of target glucose range
        post_period : str
            Length of post-intervention period to consider
        model_type : str, optional
            Type of model to use for prediction
        n_steps : int, optional
            Number of doses to evaluate in the range
            
        Returns:
        --------
        dict
            Results of optimization including optimal dose and TIR metrics
        """
        min_dose, max_dose = dose_range
        doses = np.linspace(min_dose, max_dose, n_steps)
        
        # Convert intervention_time to datetime if it's a string
        if isinstance(intervention_time, str):
            intervention_time = pd.to_datetime(intervention_time)
        
        # Get predictions for each dose
        tir_results = []
        dose_values = []
        tir_values = []
        
        for dose in doses:
            prediction = self.predict_glucose(
                pre_period_data=pre_period_data,
                intervention_time=intervention_time,
                post_period_length=post_period,
                intervention_value=dose,
                model_type=model_type
            )
            
            # Calculate time in range metrics
            total_points = len(prediction)
            in_range = ((prediction['predicted'] >= low_threshold) & 
                        (prediction['predicted'] <= high_threshold)).sum()
            below_range = (prediction['predicted'] < low_threshold).sum()
            above_range = (prediction['predicted'] > high_threshold).sum()
            
            # Calculate percentages
            tir_percentage = (in_range / total_points) * 100
            below_percentage = (below_range / total_points) * 100
            above_percentage = (above_range / total_points) * 100
            
            # Calculate mean glucose
            mean_glucose = prediction['predicted'].mean()
            
            result = {
                'dose': dose,
                'tir': tir_percentage,
                'below_range': below_percentage,
                'above_range': above_percentage,
                'mean_glucose': mean_glucose,
                'prediction': prediction
            }
            
            tir_results.append(result)
            dose_values.append(dose)
            tir_values.append(tir_percentage)
        
        # Find optimal dose (highest TIR)
        optimal_idx = np.argmax(tir_values)
        optimal_dose = dose_values[optimal_idx]
        optimal_tir = tir_values[optimal_idx]
        optimal_result = tir_results[optimal_idx]
        
        print(f"Optimal dose for maximum time in range ({low_threshold}-{high_threshold} mg/dL): {optimal_dose:.2f}u")
        print(f"Expected time in range: {optimal_tir:.1f}%")
        print(f"Expected mean glucose: {optimal_result['mean_glucose']:.1f} mg/dL")
        
        # Create figure showing dose-TIR relationship
        plt.figure(figsize=(10, 6))
        plt.plot(dose_values, tir_values, 'bo-')
        plt.axvline(x=optimal_dose, color='r', linestyle='--', label=f'Optimal: {optimal_dose:.2f}u')
        plt.xlabel('Insulin Dose (units)')
        plt.ylabel('Time in Range (%)')
        plt.title(f'Insulin Dose vs. Time in Range ({low_threshold}-{high_threshold} mg/dL)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return {
            'optimal_dose': optimal_dose,
            'optimal_tir': optimal_tir,
            'below_range': optimal_result['below_range'],
            'above_range': optimal_result['above_range'],
            'mean_glucose': optimal_result['mean_glucose'],
            'all_results': tir_results
        }
        
    def plot_time_in_range_comparison(self, pre_period_data, counterfactual_results, intervention_time,
                                    low_threshold=80, high_threshold=130, output_path=None):
        """
        Plot comparison of different doses showing time in range metrics.
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Original pre-period data
        counterfactual_results : dict
            Dictionary of prediction dataframes for different doses
        intervention_time : datetime
            Time of intervention
        low_threshold : float
            Lower bound of target glucose range
        high_threshold : float
            Upper bound of target glucose range
        output_path : str, optional
            Path to save the plot
        """
        target_col = 'glucose' if 'glucose' in pre_period_data.columns else 'response'
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        
        # Plot 1: Glucose trends for each dose
        # Plot pre-period data
        ax1.plot(pre_period_data.index, pre_period_data[target_col], 'b-', label='Historical Data')
        
        # Define a color map for different doses
        colors = ['r', 'g', 'purple', 'orange', 'c', 'm', 'gray', 'brown']
        
        # Track TIR metrics for bar chart
        doses = []
        tir_values = []
        below_values = []
        above_values = []
        labels = []
        
        # Plot each counterfactual prediction
        for i, (dose_key, prediction_df) in enumerate(counterfactual_results.items()):
            color = colors[i % len(colors)]
            
            if dose_key == 0.0 or str(dose_key) == '0.0':
                label = 'No insulin'
            else:
                label = f'Dose {dose_key}u'
                
            labels.append(label)
            
            # Plot glucose prediction
            ax1.plot(prediction_df.index, prediction_df['predicted'], 
                    color=color, linestyle='-', label=label,
                    linewidth=2)
            
            # Calculate time in range metrics
            total_points = len(prediction_df)
            in_range = ((prediction_df['predicted'] >= low_threshold) & 
                      (prediction_df['predicted'] <= high_threshold)).sum()
            below_range = (prediction_df['predicted'] < low_threshold).sum()
            above_range = (prediction_df['predicted'] > high_threshold).sum()
            
            # Calculate percentages
            tir_percentage = (in_range / total_points) * 100
            below_percentage = (below_range / total_points) * 100
            above_percentage = (above_range / total_points) * 100
            
            # Store for bar chart
            doses.append(dose_key)
            tir_values.append(tir_percentage)
            below_values.append(below_percentage)
            above_values.append(above_percentage)
        
        # Add target range
        ax1.axhline(y=high_threshold, color='red', linestyle='--', alpha=0.7, label=f'Target Range ({low_threshold}-{high_threshold} mg/dL)')
        ax1.axhline(y=low_threshold, color='red', linestyle='--', alpha=0.7)
        ax1.axhspan(low_threshold, high_threshold, alpha=0.1, color='green')
        
        # Mark intervention
        ax1.axvline(x=intervention_time, color='k', linestyle='-', label='Insulin Dose')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Glucose (mg/dL)')
        ax1.set_title('Predicted Glucose Response with Different Insulin Doses')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bar chart of time in range metrics
        width = 0.25  # width of bars
        x = np.arange(len(doses))  # label locations
        
        # Create bars
        ax2.bar(x - width, tir_values, width, label='In Range', color='green')
        ax2.bar(x, below_values, width, label='Below Range', color='blue')
        ax2.bar(x + width, above_values, width, label='Above Range', color='red')
        
        # Add labels and legend
        ax2.set_xlabel('Insulin Dose')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Time in Range Metrics by Dose')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Time-in-range comparison plot saved to {output_path}")
        else:
            plt.show()
