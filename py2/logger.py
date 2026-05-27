import json
import time
from pathlib import Path
from datetime import datetime

class SimLogger:
    def __init__(self, experiment_name, base_dir="../results"):
        """
        Initializes the logging directory and creates a RUNNING lock file.
        """
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / experiment_name / f"run_{timestamp}"
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging experiment '{experiment_name}' to: {self.run_dir}")
        
        # Create a temporary status file to indicate the run is in progress
        self.status_file = self.run_dir / "_status_RUNNING.json"
        self.log_dict({"status": "running", "start_time": timestamp}, self.status_file.name)

    def log_dict(self, data_dict, filename):
        """
        Writes a Python dictionary to a formatted JSON file.
        """
        if not filename.endswith(".json"):
            filename += ".json"
            
        filepath = self.run_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(data_dict, f, indent=4)

    def log_metrics(self, scenario_name, n, metrics_dict):
        """
        Helper method specifically for standardizing metric output names.
        """
        filename = f"metrics_{scenario_name}_N{n}.json"
        self.log_dict(metrics_dict, filename)
        
    def log_finish(self):
        """
        Renames the RUNNING status file to COMPLETED and logs total time.
        """
        if self.status_file.exists():
            # Calculate total run time
            elapsed_time = time.time() - self.start_time
            end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create the final completion dictionary
            completion_data = {
                "status": "completed",
                "end_time": end_timestamp,
                "total_duration_seconds": elapsed_time,
                "total_duration_hours": elapsed_time / 3600
            }
            
            # Define the new completed filename
            completed_file = self.run_dir / "_status_COMPLETED.json"
            
            # Write the final data and delete the old running file
            with open(completed_file, "w") as f:
                json.dump(completion_data, f, indent=4)
                
            self.status_file.unlink() # Deletes the _status_RUNNING.json file
            print(f"\nRun successfully completed in {elapsed_time/60:.2f} minutes.")