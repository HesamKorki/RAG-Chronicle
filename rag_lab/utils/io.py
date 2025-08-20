"""I/O utilities for saving and loading experiment artifacts."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data from pickle file."""
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_numpy(array: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save numpy array to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(filepath, array)


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """Load numpy array from file."""
    filepath = Path(filepath)
    
    return np.load(filepath)


def save_csv(data: Union[List[Dict], pd.DataFrame], filepath: Union[str, Path], **kwargs) -> None:
    """Save data to CSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    
    df.to_csv(filepath, index=False, **kwargs)


def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from CSV file."""
    filepath = Path(filepath)
    
    return pd.read_csv(filepath, **kwargs)


def save_jsonl(data: List[Dict], filepath: Union[str, Path]) -> None:
    """Save data to JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')


def load_jsonl(filepath: Union[str, Path]) -> List[Dict]:
    """Load data from JSONL file."""
    filepath = Path(filepath)
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return data


def save_experiment_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    experiment_name: str,
    timestamp: str
) -> None:
    """Save experiment results in a structured way."""
    output_dir = Path(output_dir)
    experiment_dir = output_dir / timestamp / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    save_json(results, experiment_dir / "results.json")
    
    # Save metrics separately if they exist
    if "metrics" in results:
        save_json(results["metrics"], experiment_dir / "metrics.json")
    
    # Save predictions if they exist
    if "predictions" in results:
        save_jsonl(results["predictions"], experiment_dir / "predictions.jsonl")
    
    # Save configuration if it exists
    if "config" in results:
        save_json(results["config"], experiment_dir / "config.json")
    
    print(f"Saved experiment results to {experiment_dir}")


def load_experiment_results(
    output_dir: Union[str, Path],
    experiment_name: str,
    timestamp: str
) -> Dict[str, Any]:
    """Load experiment results."""
    experiment_dir = Path(output_dir) / timestamp / experiment_name
    
    results = {}
    
    # Load main results
    if (experiment_dir / "results.json").exists():
        results.update(load_json(experiment_dir / "results.json"))
    
    # Load metrics
    if (experiment_dir / "metrics.json").exists():
        results["metrics"] = load_json(experiment_dir / "metrics.json")
    
    # Load predictions
    if (experiment_dir / "predictions.jsonl").exists():
        results["predictions"] = load_jsonl(experiment_dir / "predictions.jsonl")
    
    # Load configuration
    if (experiment_dir / "config.json").exists():
        results["config"] = load_json(experiment_dir / "config.json")
    
    return results


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(filepath: Union[str, Path]) -> int:
    """Get file size in bytes."""
    filepath = Path(filepath)
    return filepath.stat().st_size if filepath.exists() else 0


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def list_experiment_runs(output_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """List all experiment runs in the output directory."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        return []
    
    runs = []
    for timestamp_dir in output_dir.iterdir():
        if timestamp_dir.is_dir():
            for experiment_dir in timestamp_dir.iterdir():
                if experiment_dir.is_dir():
                    run_info = {
                        "timestamp": timestamp_dir.name,
                        "experiment": experiment_dir.name,
                        "path": experiment_dir,
                        "config_exists": (experiment_dir / "config.json").exists(),
                        "results_exists": (experiment_dir / "results.json").exists(),
                        "metrics_exists": (experiment_dir / "metrics.json").exists(),
                        "predictions_exists": (experiment_dir / "predictions.jsonl").exists(),
                    }
                    runs.append(run_info)
    
    return runs
