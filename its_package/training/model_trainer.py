import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import warnings

# Import model classes
from ..its_models.causalimpact_model import CausalImpactModel
from ..its_models.statsmodels_its import StatsmodelsITSModel
from ..data_handling.data_loader import clean_data_for_causalimpact

class ITSModelTrainer:
    """
    Trainer class for Interrupted Time Series models
    
    This class handles training, evaluation, and saving of both
    CausalImpact and StatsModels ITS models, including ensemble options.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the model trainer
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save model outputs
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model storage
        self.models = {
            "causalimpact": None,
            "statsmodels": None,
            "ensemble": None
        }
        
        # Storage for training data characteristics
        self.training_info = {}
        self.evaluation_results = {}
        
    def evaluate_prediction_accuracy(self, model, window_data, event, pre_period, post_period, model_type="causalimpact"):
        """
        Evaluate a model's prediction accuracy based only on pre-intervention data.
        
        Parameters:
        -----------
        model : CausalImpactModel or StatsmodelsITSModel
            The trained model to evaluate
        window_data : DataFrame
            Original data window containing both pre and post period data
        event : timestamp
            Time of intervention
        pre_period : list
            [start, end] of pre-intervention period
        post_period : list
            [start, end] of post-intervention period
        model_type : str
            Type of model ('causalimpact' or 'statsmodels')
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Extract pre-period data
        pre_data = window_data.loc[pre_period[0]:pre_period[1]].copy()
        
        # Extract actual post-period data for comparison
        actual_post_data = window_data.loc[post_period[0]:post_period[1]].copy()
        if actual_post_data.empty:
            raise ValueError("No post-intervention data available for evaluation")
            
        # Get target column depending on model type
        target_col = 'glucose'
        if model_type == "statsmodels" and hasattr(model, 'target_col'):
            target_col = model.target_col
        
        # Extract actual values for reference
        actual_values = actual_post_data[target_col]
        
        # Generate predictions based on pre-intervention data only
        if model_type == "causalimpact":
            # For CausalImpact, we need to use the actual model to predict
            try:
                from ..prediction.model_predictor import ITSPredictor
                
                # Create a predictor to make pure predictions
                predictor = ITSPredictor()
                predictions = predictor._predict_with_causalimpact(
                    pre_period_data=pre_data, 
                    intervention_time=event,
                    post_period_index=actual_post_data.index,
                    intervention_value=window_data.loc[event, 'insulin'] 
                        if event in window_data.index and 'insulin' in window_data.columns else None
                )
                predicted_values = predictions['predicted']
            except ImportError:
                # Fallback if the predictor module isn't available
                # Use a simplified approach
                predicted_values = model.impact.inferences.loc[actual_values.index, 'point_pred']
                
        elif model_type == "statsmodels":
            # For StatsModels, make predictions using the model coefficients
            try:
                from ..prediction.model_predictor import ITSPredictor
                
                # Create a predictor to make pure predictions  
                predictor = ITSPredictor()
                predictions = predictor._predict_with_statsmodels(
                    pre_period_data=pre_data,
                    intervention_time=event,
                    post_period_index=actual_post_data.index,
                    intervention_value=window_data.loc[event, 'insulin']
                        if event in window_data.index and 'insulin' in window_data.columns else None
                )
                predicted_values = predictions['predicted']
            except ImportError:
                # Fallback if the predictor module isn't available
                # Extract parameters from the model
                if not hasattr(model, 'params') or model.params is None:
                    raise ValueError("Model parameters not available for prediction")
                
                params = model.params
                
                # Create prediction data
                start_time = pre_data.index.min()
                pred_data = pd.DataFrame(index=actual_post_data.index)
                pred_data['time'] = (pred_data.index - start_time).total_seconds() / 60
                pred_data['post'] = 1
                pred_data['time_post'] = pred_data['time'] * pred_data['post']
                
                # Calculate predictions manually
                baseline = params.get('Intercept', 0) + params.get('time', 0) * pred_data['time']
                effect = params.get('post', 0) * pred_data['post'] + params.get('time_post', 0) * pred_data['time_post']
                predicted_values = baseline + effect
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Ensure predicted_values is a pandas Series with the right index
        if not isinstance(predicted_values, pd.Series):
            predicted_values = pd.Series(predicted_values, index=actual_values.index)
            
        # Calculate evaluation metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from math import sqrt
        
        metrics_dict = {
            'MAE': mean_absolute_error(actual_values, predicted_values),
            'MSE': mean_squared_error(actual_values, predicted_values),
            'RMSE': sqrt(mean_squared_error(actual_values, predicted_values)),
            'R2': r2_score(actual_values, predicted_values),
            'MAPE': np.mean(np.abs((actual_values - predicted_values) / (actual_values + 1e-10))) * 100
        }
        
        # Add event time and insulin dose if available
        metrics_dict['event_time'] = event
        if event in window_data.index and 'insulin' in window_data.columns:
            metrics_dict['insulin_dose'] = window_data.loc[event, 'insulin']
            
        return metrics_dict
        
    def train_models(self, data, event_times, model_types=["causalimpact", "statsmodels"],
                    pre_window='45min', post_window='30min', target_col='glucose', 
                    evaluate_prediction=True):
        """
        Train ITS models on multiple events in the dataset
        
        Parameters:
        -----------
        data : DataFrame
            Time series data containing events to model
        event_times : list
            List of timestamps corresponding to intervention events
        model_types : list, optional
            List of model types to train ('causalimpact', 'statsmodels', or both)
        pre_window : str, optional
            Time window before event for training
        post_window : str, optional
            Time window after event for evaluation
        target_col : str, optional
            Name of the target column to model
        evaluate_prediction : bool, optional
            Whether to evaluate prediction accuracy (using only pre-intervention data)
            
        Returns:
        --------
        dict
            Dictionary of trained models and evaluation metrics
        """
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if not event_times:
            raise ValueError("No event times provided")
            
        for model_type in model_types:
            if model_type not in ["causalimpact", "statsmodels"]:
                raise ValueError(f"Unknown model type: {model_type}")
                
        # Store training info
        self.training_info = {
            "pre_window": pre_window,
            "post_window": post_window,
            "target_col": target_col,
            "num_events": len(event_times),
            "data_shape": data.shape,
            "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Train models for each event
        ci_models = []
        sm_models = []
        ci_eval_metrics = []
        sm_eval_metrics = []
        ci_prediction_metrics = []  # New: for prediction-based evaluation
        sm_prediction_metrics = []  # New: for prediction-based evaluation
        
        # Process each event
        for i, event_time in enumerate(event_times):
            print(f"Training models for event {i+1}/{len(event_times)} at {event_time}")
            
            # Extract window data around the event
            window_start = event_time - pd.Timedelta(pre_window)
            window_end = event_time + pd.Timedelta(post_window)
            window_data = data.loc[window_start:window_end].copy()
            
            if window_data.empty:
                print(f"Skipping event - no data in window")
                continue
                
            # Get pre/post periods
            closest_pre = window_data.index[window_data.index <= event_time].max()
            closest_post = window_data.index[window_data.index > event_time].min()
            
            if pd.isna(closest_pre) or pd.isna(closest_post):
                print(f"Skipping event - cannot establish pre/post boundaries")
                continue
                
            pre_period = [window_data.index.min(), closest_pre]
            post_period = [closest_post, window_data.index.max()]
            
            # Train CausalImpact model if requested
            if "causalimpact" in model_types:
                try:
                    # Clean data for CausalImpact
                    ci_data = clean_data_for_causalimpact(window_data)
                    
                    # Fit model
                    ci_model = CausalImpactModel(output_dir=self.output_dir)
                    ci_model.fit(ci_data, pre_period, post_period)
                    
                    # Calculate standard evaluation metrics (fit quality)
                    metrics = self._evaluate_causalimpact_model(ci_model, post_period)
                    metrics['event_time'] = event_time
                    metrics['event_index'] = i
                    
                    # Calculate prediction-based evaluation metrics if requested
                    if evaluate_prediction:
                        pred_metrics = self.evaluate_prediction_accuracy(
                            model=ci_model,
                            window_data=ci_data,
                            event=event_time,
                            pre_period=pre_period,
                            post_period=post_period,
                            model_type="causalimpact"
                        )
                        pred_metrics['event_index'] = i
                        ci_prediction_metrics.append(pred_metrics)
                        
                        print(f"CausalImpact prediction accuracy - MAE: {pred_metrics['MAE']:.2f}, RMSE: {pred_metrics['RMSE']:.2f}")
                    
                    # Store model and metrics
                    ci_models.append(ci_model)
                    ci_eval_metrics.append(metrics)
                    
                    print(f"CausalImpact model trained - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")
                    
                except Exception as e:
                    print(f"Error training CausalImpact model: {str(e)}")
            
            # Train StatsModels ITS model if requested
            if "statsmodels" in model_types:
                try:
                    # Fit model
                    sm_model = StatsmodelsITSModel(output_dir=self.output_dir)
                    sm_model.fit(window_data, pre_period, post_period, target_col=target_col)
                    
                    # Calculate standard evaluation metrics (fit quality)
                    metrics = self._evaluate_statsmodels_model(sm_model)
                    metrics['event_time'] = event_time
                    metrics['event_index'] = i
                    
                    # Calculate prediction-based evaluation metrics if requested
                    if evaluate_prediction:
                        pred_metrics = self.evaluate_prediction_accuracy(
                            model=sm_model,
                            window_data=window_data,
                            event=event_time,
                            pre_period=pre_period,
                            post_period=post_period,
                            model_type="statsmodels"
                        )
                        pred_metrics['event_index'] = i
                        sm_prediction_metrics.append(pred_metrics)
                        
                        print(f"StatsModels prediction accuracy - MAE: {pred_metrics['MAE']:.2f}, RMSE: {pred_metrics['RMSE']:.2f}")
                    
                    # Store model and metrics
                    sm_models.append(sm_model)
                    sm_eval_metrics.append(metrics)
                    
                    print(f"StatsModels ITS model trained - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")
                    
                except Exception as e:
                    print(f"Error training StatsModels model: {str(e)}")
        
        # Aggregate model results and select best models based on metrics
        if ci_models:
            self.models["causalimpact"] = self._select_best_model(ci_models, ci_eval_metrics)
            self.evaluation_results["causalimpact"] = pd.DataFrame(ci_eval_metrics)
            
            # Store prediction-based evaluation results if available
            if ci_prediction_metrics:
                self.evaluation_results["causalimpact_prediction"] = pd.DataFrame(ci_prediction_metrics)
        
        if sm_models:
            self.models["statsmodels"] = self._select_best_model(sm_models, sm_eval_metrics)
            self.evaluation_results["statsmodels"] = pd.DataFrame(sm_eval_metrics)
            
            # Store prediction-based evaluation results if available
            if sm_prediction_metrics:
                self.evaluation_results["statsmodels_prediction"] = pd.DataFrame(sm_prediction_metrics)
        
        # Create ensemble if both model types were trained
        if self.models["causalimpact"] and self.models["statsmodels"]:
            self.models["ensemble"] = {
                "causalimpact": self.models["causalimpact"],
                "statsmodels": self.models["statsmodels"]
            }
            
            print("\nEnsemble model created from best CausalImpact and StatsModels models")
            
            # If we have prediction metrics, print summary comparison
            if ci_prediction_metrics and sm_prediction_metrics:
                ci_pred_df = pd.DataFrame(ci_prediction_metrics)
                sm_pred_df = pd.DataFrame(sm_prediction_metrics)
                
                print("\nPrediction Accuracy Comparison:")
                print(f"CausalImpact - Avg RMSE: {ci_pred_df['RMSE'].mean():.2f}, Avg MAE: {ci_pred_df['MAE'].mean():.2f}")
                print(f"StatsModels  - Avg RMSE: {sm_pred_df['RMSE'].mean():.2f}, Avg MAE: {sm_pred_df['MAE'].mean():.2f}")
                
                # Select best model based on prediction accuracy if data available
                if not ci_pred_df.empty and not sm_pred_df.empty:
                    ci_rmse = ci_pred_df['RMSE'].mean()
                    sm_rmse = sm_pred_df['RMSE'].mean()
                    better_model = "CausalImpact" if ci_rmse <= sm_rmse else "StatsModels"
                    print(f"\nBased on prediction accuracy, {better_model} performs better")
        
        # Return summary of training
        return self._get_training_summary()
    
    def _select_best_model(self, models, metrics_list):
        """
        Select the best model based on evaluation metrics
        Default strategy: select model with lowest RMSE
        """
        if not models:
            return None
            
        # Convert metrics to DataFrame for easier selection
        metrics_df = pd.DataFrame(metrics_list)
        
        # Find index of model with lowest RMSE
        best_idx = metrics_df['RMSE'].argmin()
        
        print(f"Selected best model (index {best_idx}) with RMSE: {metrics_df.iloc[best_idx]['RMSE']:.2f}")
        
        return models[best_idx]
    
    def _evaluate_causalimpact_model(self, model, post_period):
        """Calculate evaluation metrics for CausalImpact model"""
        impact = model.impact
        
        # Extract actual and predicted values for post-period
        post_data = impact.inferences.loc[impact.inferences.index >= post_period[0]]
        actual = post_data['response']
        predicted = post_data['point_pred']
        
        return {
            'MAE': mean_absolute_error(actual, predicted),
            'MSE': mean_squared_error(actual, predicted),
            'RMSE': sqrt(mean_squared_error(actual, predicted)),
            'R2': r2_score(actual, predicted),
            'MAPE': np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        }
    
    def _evaluate_statsmodels_model(self, model):
        """Calculate evaluation metrics for StatsModels ITS model"""
        # Get actual and fitted values
        actual = model.data[model.target_col]
        fitted = model.results.fittedvalues
        
        return {
            'MAE': mean_absolute_error(actual, fitted),
            'MSE': mean_squared_error(actual, fitted),
            'RMSE': sqrt(mean_squared_error(actual, fitted)),
            'R2': model.results.rsquared,
            'MAPE': np.mean(np.abs((actual - fitted) / (actual + 1e-10))) * 100
        }
    
    def save_models(self, base_filename=None):
        """
        Save all trained models to files
        
        Parameters:
        -----------
        base_filename : str, optional
            Base name for saved model files
            
        Returns:
        --------
        dict
            Dictionary with paths to saved model files
        """
        if not self.output_dir:
            self.output_dir = "."
            
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"its_model_{timestamp}"
            
        saved_paths = {}
        
        # Save individual models
        for model_type, model in self.models.items():
            if model is None:
                continue
                
            if model_type == "causalimpact":
                filename = f"{base_filename}_causalimpact.pkl"
                path = os.path.join(self.output_dir, filename)
                model.save_model(path)
                saved_paths["causalimpact"] = path
                
            elif model_type == "statsmodels":
                filename = f"{base_filename}_statsmodels.pkl"
                path = os.path.join(self.output_dir, filename)
                model.save_model(path)
                saved_paths["statsmodels"] = path
                
            elif model_type == "ensemble":
                # For ensemble, save a reference file with paths to component models
                filename = f"{base_filename}_ensemble.json"
                path = os.path.join(self.output_dir, filename)
                
                ensemble_info = {
                    "causalimpact_model": saved_paths.get("causalimpact"),
                    "statsmodels_model": saved_paths.get("statsmodels"),
                    "training_info": self.training_info,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(path, 'wb') as f:
                    pickle.dump(ensemble_info, f)
                
                saved_paths["ensemble"] = path
                
        # Save evaluation results
        if self.evaluation_results:
            eval_path = os.path.join(self.output_dir, f"{base_filename}_evaluation.pkl")
            with open(eval_path, 'wb') as f:
                pickle.dump(self.evaluation_results, f)
            saved_paths["evaluation"] = eval_path
                
        return saved_paths
    
    def _get_training_summary(self):
        """Get a summary of the training results"""
        summary = {
            "training_info": self.training_info,
            "models_trained": {}
        }
        
        # Add model-specific summaries
        for model_type, model in self.models.items():
            if model is None:
                continue
                
            if model_type == "causalimpact":
                summary["models_trained"]["causalimpact"] = {
                    "trained": True,
                    "evaluation": self.evaluation_results.get("causalimpact", pd.DataFrame()).mean().to_dict()
                }
                
            elif model_type == "statsmodels":
                summary["models_trained"]["statsmodels"] = {
                    "trained": True,
                    "evaluation": self.evaluation_results.get("statsmodels", pd.DataFrame()).mean().to_dict()
                }
                
            elif model_type == "ensemble" and model is not None:
                summary["models_trained"]["ensemble"] = {
                    "trained": True,
                    "components": ["causalimpact", "statsmodels"]
                }
                
        return summary
    
    def plot_model_comparison(self, event_data=None, event_time=None, figsize=(12, 8)):
        """
        Plot comparison of model predictions
        
        Parameters:
        -----------
        event_data : DataFrame, optional
            Data surrounding an event for prediction comparison
        event_time : timestamp, optional
            Time of intervention event
        figsize : tuple, optional
            Size of the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with plot
        """
        if event_data is None or event_time is None:
            raise ValueError("Both event_data and event_time must be provided")
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual data
        target_col = self.training_info.get("target_col", "glucose")
        ax.plot(event_data.index, event_data[target_col], 'b-', label="Actual Data", linewidth=2)
        
        # Plot vertical line for event time
        ax.axvline(x=event_time, color='k', linestyle='--', label="Intervention")
        
        # Generate predictions from each model
        if self.models["causalimpact"]:
            # Get CausalImpact predictions
            try:
                ci_preds = self._get_causalimpact_predictions(event_data, event_time)
                ax.plot(ci_preds.index, ci_preds, 'r-', label="CausalImpact Prediction", alpha=0.7)
            except Exception as e:
                print(f"Error getting CausalImpact predictions: {str(e)}")
            
        if self.models["statsmodels"]:
            # Get StatsModels predictions
            try:
                sm_preds = self._get_statsmodels_predictions(event_data, event_time)
                ax.plot(sm_preds.index, sm_preds, 'g-', label="StatsModels Prediction", alpha=0.7)
            except Exception as e:
                print(f"Error getting StatsModels predictions: {str(e)}")
            
        if self.models["causalimpact"] and self.models["statsmodels"]:
            # Create ensemble prediction (average)
            try:
                ensemble_preds = (ci_preds + sm_preds) / 2
                ax.plot(ensemble_preds.index, ensemble_preds, 'purple', 
                        label="Ensemble Prediction", linestyle='-', linewidth=2, alpha=0.8)
            except Exception as e:
                print(f"Error creating ensemble predictions: {str(e)}")
            
        # Finalize plot
        ax.set_title("Model Predictions Comparison")
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _get_causalimpact_predictions(self, event_data, event_time):
        """Get predictions from CausalImpact model"""
        model = self.models["causalimpact"]
        if model is None:
            raise ValueError("CausalImpact model not trained")
            
        # Return the point predictions
        return model.impact.inferences['point_pred']
    
    def _get_statsmodels_predictions(self, event_data, event_time):
        """Get predictions from StatsModels ITS model"""
        model = self.models["statsmodels"]
        if model is None:
            raise ValueError("StatsModels model not trained")
            
        # Return the fitted values
        return pd.Series(model.results.fittedvalues, index=model.data['index'])