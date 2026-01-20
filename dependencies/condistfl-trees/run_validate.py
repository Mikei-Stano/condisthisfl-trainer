#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run cross-site validation after training.

Usage:
    python run_validate.py -w workspace -o cross_site_validate.json
"""

import argparse
import json
from pathlib import Path


def run_validation(workspace_dir: Path, output_file: str):
    """
    Parse and display cross-site validation results.
    
    Args:
        workspace_dir: Workspace directory from training
        output_file: Output JSON file name
    """
    # NVFlare simulator stores results in server/simulate_job/cross_site_val/
    results_file = workspace_dir / "server" / "simulate_job" / "cross_site_val" / output_file
    
    if not results_file.exists():
        print(f"Error: Results file {results_file} not found!")
        print("Make sure training has completed with cross-site validation.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("=" * 80)
    print("CROSS-SITE VALIDATION RESULTS")
    print("=" * 80)
    print()
    
    # NVFlare format: {site_name: {model_name: {metrics}}}
    # Reorganize by model for clearer comparison
    all_models = {}
    for site_name, site_results in results.items():
        for model_name, metrics in site_results.items():
            if model_name not in all_models:
                all_models[model_name] = {}
            all_models[model_name][site_name] = metrics
    
    # Display results for each model
    for model_name, site_metrics in all_models.items():
        print(f"Model: {model_name}")
        print("-" * 80)
        
        # Average metrics across sites
        all_metric_names = set()
        for metrics in site_metrics.values():
            all_metric_names.update(metrics.keys())
        
        avg_metrics = {}
        for metric_name in all_metric_names:
            values = [m[metric_name] for m in site_metrics.values() if metric_name in m]
            avg_metrics[metric_name] = sum(values) / len(values) if values else 0
        
        print(f"  Average across all sites:")
        for metric_name, value in sorted(avg_metrics.items()):
            print(f"    {metric_name}: {value:.4f}")
        
        print(f"\n  Per-site breakdown:")
        for site_name, metrics in sorted(site_metrics.items()):
            print(f"    {site_name}:")
            for metric_name, value in sorted(metrics.items()):
                print(f"      {metric_name}: {value:.4f}")
        print()
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run cross-site validation")
    parser.add_argument("-w", "--workspace", type=str, default="workspace",
                        help="Workspace directory")
    parser.add_argument("-o", "--output", type=str, default="cross_val_results.json",
                        help="Output JSON file name")
    
    args = parser.parse_args()
    
    workspace_dir = Path(args.workspace)
    
    if not workspace_dir.exists():
        print(f"Error: Workspace directory {workspace_dir} does not exist!")
        return
    
    run_validation(workspace_dir, args.output)


if __name__ == "__main__":
    main()
