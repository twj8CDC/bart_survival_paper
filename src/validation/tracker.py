"""
Structured File-Based Experiment Tracker (Replacing MLflow).
Stores run metadata, replicate-level metrics, causal estimands, and summary aggregations cleanly.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

class BenchmarkTracker:
    def __init__(self, experiment_name: str, base_dir: Optional[str] = None):
        self.experiment_name = experiment_name
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent.parent / "results" / experiment_name
        else:
            self.base_dir = Path(base_dir) / experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        
    def get_run_dir(self, scenario_id: str, n: int, preset: str) -> Path:
        run_name = f"{scenario_id}_N{n}_{preset}"
        run_dir = self.base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
        
    def load_replicates(self, scenario_id: str, n: int, preset: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        run_dir = self.get_run_dir(scenario_id, n, preset)
        rep_file = run_dir / "replicate_metrics.json"
        causal_file = run_dir / "causal_effects.json"
        reps, causal = [], []
        if rep_file.exists():
            try:
                with open(rep_file, "r") as f:
                    reps = json.load(f)
            except Exception:
                pass
        if causal_file.exists():
            try:
                with open(causal_file, "r") as f:
                    causal = json.load(f)
            except Exception:
                pass
        return reps, causal
        
    def log_run_start(self, scenario_id: str, n: int, preset: str, scenario_cfg: Dict[str, Any], model_cfg: Dict[str, Any]):
        run_dir = self.get_run_dir(scenario_id, n, preset)
        meta = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_cfg.get("name"),
            "category": scenario_cfg.get("category"),
            "n": n,
            "preset": preset,
            "scenario_config": scenario_cfg,
            "model_config": model_cfg,
            "start_time": datetime.now().isoformat(),
            "status": "RUNNING",
        }
        with open(run_dir / "run_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
            
    def log_replicates(
        self,
        scenario_id: str,
        n: int,
        preset: str,
        replicate_records: List[Dict[str, Any]],
        causal_records: Optional[List[Dict[str, Any]]] = None
    ):
        run_dir = self.get_run_dir(scenario_id, n, preset)
        
        # Save metrics DataFrame
        df_reps = pd.DataFrame(replicate_records)
        df_reps.to_json(run_dir / "replicate_metrics.json", orient="records", indent=2)
        try:
            df_reps.to_parquet(run_dir / "replicate_metrics.parquet", index=False)
        except Exception:
            pass
            
        # Save causal DataFrame
        if causal_records:
            df_causal = pd.DataFrame(causal_records)
            df_causal.to_json(run_dir / "causal_effects.json", orient="records", indent=2)
            try:
                df_causal.to_parquet(run_dir / "causal_effects.parquet", index=False)
            except Exception:
                pass
                
        # Generate summary aggregation
        numeric_cols = [c for c in df_reps.columns if c not in ["seed", "model", "scenario_id", "n", "preset"]]
        summary = {}
        for col in numeric_cols:
            vals = pd.to_numeric(df_reps[col], errors="coerce").dropna()
            if len(vals) > 0:
                summary[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if len(vals) > 1 else 0.0,
                    "median": float(vals.median()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
                
        with open(run_dir / "summary_report.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        # Update metadata status to COMPLETED
        meta_file = run_dir / "run_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
            meta["status"] = "COMPLETED"
            meta["completion_time"] = datetime.now().isoformat()
            meta["duration_seconds"] = time.time() - self.start_time
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)
                
        print(f"Logged {len(replicate_records)} replicate records to {run_dir}")

