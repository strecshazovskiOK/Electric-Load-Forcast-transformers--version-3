# evaluation/visualization.py
from typing import List, Dict, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

from .results import EvaluationResult, PredictionComparison

class EvaluationVisualizer:
    """Creates interactive visualizations for evaluation results."""

    @staticmethod
    def plot_predictions_vs_actuals(
            result: EvaluationResult,
            variable_idx: Optional[int] = None,
            window_size: Optional[int] = None
    ) -> go.Figure:
        """
        Plot predicted vs actual values.

        Args:
            result: Evaluation results
            variable_idx: Index of variable to plot (for multi-variable predictions)
            window_size: Rolling window size for smoother visualization
        """
        # Prepare data
        data = pd.DataFrame([
            {
                'timestamp': comp.timestamp,
                'predicted': comp.predicted[variable_idx] if variable_idx is not None else comp.predicted,
                'actual': comp.actual[variable_idx] if variable_idx is not None else comp.actual
            }
            for comp in result.comparisons
        ])

        if window_size:
            data['predicted'] = data['predicted'].rolling(window_size).mean()
            data['actual'] = data['actual'].rolling(window_size).mean()

        # Create figure
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['actual'],
            name='Actual',
            mode='lines',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['predicted'],
            name='Predicted',
            mode='lines',
            line=dict(color='red')
        ))

        # Update layout
        fig.update_layout(
            title='Predictions vs Actuals',
            xaxis_title='Timestamp',
            yaxis_title='Value',
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def plot_error_distribution(result: EvaluationResult) -> go.Figure:
        """Plot distribution of prediction errors."""
        # Calculate errors
        errors = [
            comp.predicted - comp.actual
            for comp in result.comparisons
        ]

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=50,
            name='Error Distribution'
        ))

        fig.update_layout(
            title='Prediction Error Distribution',
            xaxis_title='Error',
            yaxis_title='Count',
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_metrics_over_time(
            result: EvaluationResult,
            window_size: int = 24
    ) -> go.Figure:
        """Plot rolling metrics over time."""
        # Calculate rolling metrics
        data = []
        for i in range(len(result.comparisons)):
            window = result.comparisons[max(0, i-window_size):i+1]
            if len(window) >= window_size:
                metrics = {
                    'timestamp': window[-1].timestamp,
                    'mape': calculate_rolling_mape(window),
                    'mae': calculate_rolling_mae(window),
                    'rmse': calculate_rolling_rmse(window)
                }
                data.append(metrics)

        df = pd.DataFrame(data)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['mape'],
                name="MAPE (%)",
                line=dict(color='blue')
            ),
            secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['mae'],
                name="MAE",
                line=dict(color='red')
            ),
            secondary_y=False
        )

        # Update layout
        fig.update_layout(
            title='Rolling Metrics Over Time',
            hovermode='x unified'
        )

        fig.update_yaxes(title_text="MAE", secondary_y=False)
        fig.update_yaxes(title_text="MAPE (%)", secondary_y=True)

        return fig

    @staticmethod
    def plot_residuals(result: EvaluationResult) -> go.Figure:
        """Plot residuals analysis."""
        # Prepare data
        residuals = []
        actuals = []
        for comp in result.comparisons:
            residuals.append(comp.predicted - comp.actual)
            actuals.append(comp.actual)

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=actuals,
            y=residuals,
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6
            ),
            name='Residuals'
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")

        fig.update_layout(
            title='Residuals vs Actual Values',
            xaxis_title='Actual Values',
            yaxis_title='Residuals',
            showlegend=False
        )

        return fig

    @staticmethod
    def create_evaluation_dashboard(result: EvaluationResult) -> go.Figure:
        """Create a comprehensive evaluation dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Predictions vs Actuals',
                'Error Distribution',
                'Rolling Metrics',
                'Residuals Analysis'
            )
        )

        # Add predictions vs actuals
        pred_vs_actual = EvaluationVisualizer.plot_predictions_vs_actuals(result)
        fig.add_trace(
            pred_vs_actual.data[0],
            row=1, col=1
        )
        fig.add_trace(
            pred_vs_actual.data[1],
            row=1, col=1
        )

        # Add error distribution
        error_dist = EvaluationVisualizer.plot_error_distribution(result)
        fig.add_trace(
            error_dist.data[0],
            row=1, col=2
        )

        # Add metrics over time
        metrics = EvaluationVisualizer.plot_metrics_over_time(result)
        fig.add_trace(
            metrics.data[0],
            row=2, col=1
        )

        # Add residuals
        residuals = EvaluationVisualizer.plot_residuals(result)
        fig.add_trace(
            residuals.data[0],
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            showlegend=False,
            title_text="Model Evaluation Dashboard"
        )

        return fig

# Helper functions for rolling metrics
def calculate_rolling_mape(window: List[PredictionComparison]) -> float:
    """Calculate MAPE for a window of predictions."""
    actual = [comp.actual for comp in window]
    pred = [comp.predicted for comp in window]
    return float(np.mean(np.abs((np.array(actual) - np.array(pred)) / np.array(actual))) * 100)

def calculate_rolling_mae(window: List[PredictionComparison]) -> float:
    """Calculate MAE for a window of predictions."""
    actual = [comp.actual for comp in window]
    pred = [comp.predicted for comp in window]
    return float(np.mean(np.abs(np.array(actual) - np.array(pred))))

def calculate_rolling_rmse(window: List[PredictionComparison]) -> float:
    """Calculate RMSE for a window of predictions."""
    actual = [comp.actual for comp in window]
    pred = [comp.predicted for comp in window]
    return float(np.sqrt(np.mean((np.array(actual) - np.array(pred)) ** 2)))