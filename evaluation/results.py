# evaluation/results.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

@dataclass
class PredictionComparison:
    """Stores prediction and actual values for a specific timestamp."""
    timestamp: datetime
    predicted: np.ndarray
    actual: np.ndarray

    def serialize(self) -> Dict:
        """Serialize comparison data."""
        # Handle different timestamp types
        if isinstance(self.timestamp, (int, np.integer)):
            # Convert numpy integer to datetime using pandas
            ts = pd.Timestamp.fromtimestamp(float(self.timestamp))
        elif isinstance(self.timestamp, np.datetime64):
            ts = pd.Timestamp(self.timestamp).to_pydatetime()
        else:
            ts = self.timestamp

        return {
            'timestamp': ts.isoformat(),
            'predicted': self.predicted.tolist(),
            'actual': self.actual.tolist()
        }
        
@dataclass
class EvaluationResult:
    """Stores complete evaluation results."""

    # Overall metrics
    total_metrics: Dict[str, float]

    # Per-variable metrics
    variable_metrics: Dict[str, Dict[str, float]]

    # Detailed comparisons
    comparisons: List[PredictionComparison]

    # Additional metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def serialize(self) -> Dict:
        """Serialize evaluation results."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_metrics': self.total_metrics,
            'variable_metrics': self.variable_metrics,
            'comparisons': [comp.serialize() for comp in self.comparisons],
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EvaluationResult':
        """Create EvaluationResult from serialized data."""
        return cls(
            total_metrics=data['total_metrics'],
            variable_metrics=data['variable_metrics'],
            comparisons=[
                PredictionComparison(
                    timestamp=datetime.fromisoformat(comp['timestamp']),
                    predicted=np.array(comp['predicted']),
                    actual=np.array(comp['actual'])
                )
                for comp in data['comparisons']
            ],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )