"""
I/O utilities for data persistence and loading.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data as JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def save_experiment(experiment_data: Dict[str, Any], artifacts_dir: str = "artifacts") -> str:
    """Save experiment results to artifacts directory."""
    ensure_dir(artifacts_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_{timestamp}.json"
    filepath = os.path.join(artifacts_dir, filename)
    
    # Add metadata
    experiment_data['timestamp'] = timestamp
    experiment_data['saved_at'] = datetime.now().isoformat()
    
    save_json(experiment_data, filepath)
    return filepath

def list_experiments(artifacts_dir: str = "artifacts") -> list:
    """List available experiment files."""
    if not os.path.exists(artifacts_dir):
        return []
    
    experiments = []
    for filename in os.listdir(artifacts_dir):
        if filename.startswith('experiment_') and filename.endswith('.json'):
            filepath = os.path.join(artifacts_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    experiments.append({
                        'filename': filename,
                        'timestamp': data.get('timestamp', ''),
                        'saved_at': data.get('saved_at', ''),
                        'symbols': data.get('symbols', []),
                        'metrics': data.get('metrics', {})
                    })
            except (json.JSONDecodeError, KeyError):
                continue
    
    return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)
