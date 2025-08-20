"""Utility functions for RAG experiments."""

from .text import (
    normalize_text,
    tokenize_text,
    extract_ngrams,
    extract_ngrams_range,
    remove_stopwords,
    get_common_stopwords,
    clean_text_for_boolean,
    extract_entities,
    compute_text_similarity,
    truncate_text
)

from .seeds import (
    set_seed,
    get_seed_from_config,
    seed_worker,
    set_deterministic_mode,
    reset_seeds
)

from .timing import (
    Timer,
    timer,
    PerformanceTracker,
    measure_time,
    format_time
)

from .io import (
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    save_numpy,
    load_numpy,
    save_csv,
    load_csv,
    save_jsonl,
    load_jsonl,
    save_experiment_results,
    load_experiment_results,
    ensure_directory,
    get_file_size,
    format_file_size,
    list_experiment_runs
)

__all__ = [
    # Text utilities
    "normalize_text",
    "tokenize_text",
    "extract_ngrams",
    "extract_ngrams_range", 
    "remove_stopwords",
    "get_common_stopwords",
    "clean_text_for_boolean",
    "extract_entities",
    "compute_text_similarity",
    "truncate_text",
    
    # Seed utilities
    "set_seed",
    "get_seed_from_config",
    "seed_worker",
    "set_deterministic_mode",
    "reset_seeds",
    
    # Timing utilities
    "Timer",
    "timer",
    "PerformanceTracker",
    "measure_time",
    "format_time",
    
    # I/O utilities
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "save_numpy",
    "load_numpy",
    "save_csv",
    "load_csv",
    "save_jsonl",
    "load_jsonl",
    "save_experiment_results",
    "load_experiment_results",
    "ensure_directory",
    "get_file_size",
    "format_file_size",
    "list_experiment_runs"
]
