"""
Extracts baseline r-BART (and p-BART, CPH, KM) results from previous MLflow runs.
Preserves historical 1,000-iteration results with metadata on evaluated quantiles.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List

def parse_mlruns():
    base_dir = Path("/Users/jacobtiegs/projects/bart_val_online/bart_survival_paper_1000")
    mlrun_dirs = [base_dir / "mlruns", base_dir / "mlruns2", base_dir / "mlruns3"]
    
    historical_results = {}
    
    for mlrun in mlrun_dirs:
        if not mlrun.exists():
            continue
        for root, dirs, files in os.walk(mlrun):
            # Look for artifact directories
            if "artifacts" in root:
                # Find scenario file if present
                scenario_files = [f for f in files if "scenario" in f and f.endswith(".json")]
                r_files = [f for f in files if "met_r" in f and f.endswith(".json")]
                
                if r_files:
                    for rf in r_files:
                        # e.g. 1_1_400_met_r.json or cmplx1_400_met_r.json
                        prefix = rf.replace("_met_r.json", "")
                        r_path = Path(root) / rf
                        
                        try:
                            with open(r_path, "r") as f:
                                r_data = json.load(f)
                        except Exception:
                            continue
                            
                        # Try to load scenario data if exists
                        scen_data = None
                        scen_file = Path(root) / f"{prefix}_scenario.json"
                        if scen_file.exists():
                            try:
                                with open(scen_file, "r") as f:
                                    scen_data = json.load(f)
                            except Exception:
                                pass
                                
                        # Try to load time info if exists
                        cens_data = None
                        cens_file = Path(root) / f"{prefix}_cens.json"
                        if cens_file.exists():
                            try:
                                with open(cens_file, "r") as f:
                                    cens_data = json.load(f)
                            except Exception:
                                pass
                                
                        key = f"{mlrun.name}_{prefix}"
                        historical_results[key] = {
                            "source_mlrun": mlrun.name,
                            "prefix": prefix,
                            "r_metrics": r_data,
                            "scenario": scen_data,
                            "cens": cens_data,
                            "raw_artifact_dir": root,
                        }
                        
    out_file = base_dir / "results" / "historical_rbart_baseline.json"
    with open(out_file, "w") as f:
        json.dump(historical_results, f, indent=2)
        
    print(f"Successfully extracted {len(historical_results)} historical runs to {out_file}")
    for k in sorted(historical_results.keys()):
        print(f"  - {k}")

if __name__ == "__main__":
    parse_mlruns()

