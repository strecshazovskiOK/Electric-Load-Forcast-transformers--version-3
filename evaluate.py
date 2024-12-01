# evaluate.py
import argparse
from pathlib import Path
import torch
from typing import Union, Optional
from evaluation.visualization import EvaluationVisualizer

def evaluate_model(experiment_path: str, output_path: Optional[str] = None) -> None:
    """Evaluate a trained model and generate visualizations."""
    
    # Load experiment results
    experiment = torch.load(experiment_path)
    
    # Create visualizer
    visualizer = EvaluationVisualizer()
    
    # Create evaluation dashboard
    print("Creating evaluation dashboard...")
    dashboard = visualizer.create_evaluation_dashboard(experiment.evaluation)
    
    # Save or show results
    if output_path:
        dashboard.write_html(output_path)
        print(f"Dashboard saved to {output_path}")
    else:
        dashboard.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate transformer model predictions')
    parser.add_argument('--experiment', type=str, required=True, help='Path to the experiment file')
    parser.add_argument('--output', type=str, help='Path to save the evaluation dashboard')
    args = parser.parse_args()
    
    evaluate_model(args.experiment, args.output)

if __name__ == '__main__':
    main()