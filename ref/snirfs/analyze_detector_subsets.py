#!/usr/bin/env python3
"""
Script to analyze which specific detectors from each module in layout.json
are present in the SNIRF files.
"""
import os
import json
import numpy as np
import h5py
from pathlib import Path
import argparse

def extract_snirf_positions(file_path):
    """Extract detector positions from a SNIRF file."""
    try:
        with h5py.File(file_path, 'r') as snirf_file:
            if 'nirs' not in snirf_file:
                return {"error": "No 'nirs' group found", "detectors": []}
            
            nirs = snirf_file['nirs']
            
            if 'probe' not in nirs:
                return {"error": "No 'probe' group found", "detectors": []}
            
            probe = nirs['probe']
            
            # Extract detector positions
            detectors = []
            if 'detectorPos3D' in probe:
                detector_pos = probe['detectorPos3D'][:]
                for i, pos in enumerate(detector_pos):
                    detectors.append({
                        "index": i + 1,
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2])]
                    })
            elif 'detectorPos' in probe:
                detector_pos = probe['detectorPos'][:]
                for i, pos in enumerate(detector_pos):
                    detectors.append({
                        "index": i + 1,
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0]
                    })
            
            return {"detectors": detectors}
    
    except Exception as e:
        return {"error": str(e), "detectors": []}

def load_layout_json(file_path):
    """Load detector positions from layout.json."""
    with open(file_path, 'r') as f:
        layout_data = json.load(f)
    
    # Extract detector positions
    detector_modules = []
    for module_idx, module in enumerate(layout_data.get("detector_locations", [])):
        module_detectors = []
        for pos_idx, pos in enumerate(module):
            module_detectors.append({
                "module_idx": module_idx + 1,
                "pos_idx": pos_idx + 1,
                "global_idx": len([d for m in detector_modules for d in m]) + pos_idx + 1,
                "pos": pos
            })
        detector_modules.append(module_detectors)
    
    return detector_modules

def is_position_match(pos1, pos2, tolerance=1e-3):
    """Check if two positions match within tolerance."""
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(pos1, pos2)])) < tolerance

def find_detector_matches(snirf_detectors, layout_modules):
    """Find which detectors from each module are present in SNIRF files."""
    module_coverage = []
    
    for module_idx, module in enumerate(layout_modules):
        module_matches = []
        
        for detector in module:
            # Check if this detector position exists in SNIRF
            match_found = False
            for snirf_detector in snirf_detectors:
                if is_position_match(detector["pos"], snirf_detector["pos"]):
                    match_found = True
                    module_matches.append({
                        "layout_module": detector["module_idx"],
                        "layout_pos_idx": detector["pos_idx"],
                        "layout_global_idx": detector["global_idx"],
                        "snirf_idx": snirf_detector["index"]
                    })
                    break
        
        # Calculate coverage statistics for this module
        module_coverage.append({
            "module_idx": module_idx + 1,
            "total_detectors": len(module),
            "matched_detectors": len(module_matches),
            "coverage_percent": (len(module_matches) / len(module)) * 100,
            "matches": module_matches,
            "detector_indices_present": [m["layout_pos_idx"] for m in module_matches]
        })
    
    return module_coverage

def generate_report(coverage_data, snirf_file):
    """Generate a readable report of which detectors from each module are present."""
    total_modules = len(coverage_data)
    modules_with_matches = sum(1 for m in coverage_data if m["matched_detectors"] > 0)
    
    report = f"# Detector Module Coverage Analysis: {snirf_file}\n\n"
    
    # Overall statistics
    report += f"## Overview\n\n"
    report += f"- **Modules with any detectors present**: {modules_with_matches}/{total_modules}\n"
    report += f"- **Modules with complete coverage**: {sum(1 for m in coverage_data if m['coverage_percent'] == 100)}/{total_modules}\n"
    report += f"- **Modules with partial coverage**: {sum(1 for m in coverage_data if 0 < m['coverage_percent'] < 100)}/{total_modules}\n"
    report += f"- **Modules with no coverage**: {sum(1 for m in coverage_data if m['coverage_percent'] == 0)}/{total_modules}\n\n"
    
    # Add a summary of which detector positions tend to be included
    position_frequency = {i+1: 0 for i in range(7)}  # Assuming 7 detectors per module
    
    for module in coverage_data:
        for pos_idx in module["detector_indices_present"]:
            position_frequency[pos_idx] += 1
    
    report += "## Detector Position Frequency\n\n"
    report += "Which detector positions within modules are most commonly included:\n\n"
    report += "| Position | Count | Percentage |\n"
    report += "|----------|-------|------------|\n"
    
    modules_with_coverage = sum(1 for m in coverage_data if m["matched_detectors"] > 0)
    for pos, count in position_frequency.items():
        if modules_with_coverage > 0:
            percentage = (count / modules_with_coverage) * 100
        else:
            percentage = 0
        report += f"| {pos} | {count} | {percentage:.1f}% |\n"
    
    report += "\n"
    
    # Per-module details
    report += "## Module Details\n\n"
    
    for module in coverage_data:
        if module["matched_detectors"] > 0:
            report += f"### Module {module['module_idx']}\n\n"
            report += f"- Detectors present: {module['matched_detectors']}/{module['total_detectors']} ({module['coverage_percent']:.1f}%)\n"
            report += f"- Positions included: {', '.join(str(idx) for idx in sorted(module['detector_indices_present']))}\n"
            
            if module["matched_detectors"] < module["total_detectors"]:
                missing = [i for i in range(1, module["total_detectors"] + 1) if i not in module["detector_indices_present"]]
                report += f"- Positions excluded: {', '.join(str(idx) for idx in missing)}\n"
            
            report += "\n"
    
    # Common patterns
    report += "## Pattern Analysis\n\n"
    
    # Look for modules with the same pattern of detectors
    patterns = {}
    for module in coverage_data:
        pattern_key = tuple(sorted(module["detector_indices_present"]))
        if pattern_key not in patterns:
            patterns[pattern_key] = []
        patterns[pattern_key].append(module["module_idx"])
    
    report += "Modules with identical detector coverage patterns:\n\n"
    
    for pattern, modules in sorted(patterns.items(), key=lambda x: (len(x[0]), x[0])):
        if not pattern:  # Skip empty patterns (no detectors)
            continue
        
        pattern_str = ", ".join(str(p) for p in pattern)
        modules_str = ", ".join(str(m) for m in modules)
        report += f"- Pattern [{pattern_str}] found in modules: {modules_str}\n"
    
    return report

def main():
    """Main function to analyze which detector positions are included in SNIRF files."""
    parser = argparse.ArgumentParser(description='Analyze which detector positions are included in SNIRF files.')
    parser.add_argument('--snirf', default=None, help='Path to specific SNIRF file to analyze')
    parser.add_argument('--layout', default="/Users/clem/Desktop/code/data-pipeline/ref/layout.json", 
                        help='Path to layout.json file')
    args = parser.parse_args()
    
    # Load layout.json detector modules
    layout_modules = load_layout_json(args.layout)
    print(f"Loaded {len(layout_modules)} detector modules from layout.json")
    
    # Find SNIRF files to process
    if args.snirf:
        snirf_files = [Path(args.snirf)]
    else:
        snirf_dir = "/Users/clem/Desktop/code/data-pipeline/ref/snirfs"
        snirf_files = list(Path(snirf_dir).glob("*.snirf"))
    
    print(f"Found {len(snirf_files)} SNIRF files for analysis...")
    
    for snirf_file in snirf_files:
        print(f"Analyzing {snirf_file.name}...")
        
        # Extract detector positions from SNIRF file
        snirf_data = extract_snirf_positions(snirf_file)
        
        if "error" in snirf_data and snirf_data["error"]:
            print(f"Error processing {snirf_file.name}: {snirf_data['error']}")
            continue
        
        # Find which detectors from each module are present
        coverage_data = find_detector_matches(snirf_data["detectors"], layout_modules)
        
        # Generate and save report
        report = generate_report(coverage_data, snirf_file.name)
        output_dir = snirf_file.parent
        report_file = output_dir / f"detector_coverage_{snirf_file.stem}.md"
        
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"Detector coverage report written to: {report_file}")
        
        # Save raw coverage data as JSON for further analysis
        json_file = output_dir / f"detector_coverage_{snirf_file.stem}.json"
        with open(json_file, "w") as f:
            # Convert to a JSON-serializable format
            serializable_data = []
            for module in coverage_data:
                module_copy = module.copy()
                serializable_data.append(module_copy)
            
            json.dump(serializable_data, f, indent=2)
        
        print(f"Coverage data saved to: {json_file}")
    
    # Combine reports if multiple files
    if len(snirf_files) > 1:
        combine_reports(Path(snirf_files[0]).parent)

def combine_reports(output_dir):
    """Combine all detector coverage reports into a single summary file."""
    report_files = list(output_dir.glob("detector_coverage_*.md"))
    
    combined_report = "# Detector Coverage Analysis Summary\n\n"
    combined_report += f"Analysis of detector module coverage for {len(report_files)} SNIRF files\n\n"
    combined_report += "## Table of Contents\n\n"
    
    for report_file in report_files:
        file_name = report_file.name.replace("detector_coverage_", "").replace(".md", "")
        combined_report += f"- [{file_name}](#{file_name.lower().replace('.', '').replace('_', '-')})\n"
    
    combined_report += "\n"
    
    for report_file in sorted(report_files):
        with open(report_file, "r") as f:
            content = f.read()
        
        combined_report += content + "\n---\n\n"
    
    # Write combined report
    combined_report_path = output_dir / "detector_coverage_summary.md"
    with open(combined_report_path, "w") as f:
        f.write(combined_report)
    
    print(f"Combined summary written to: {combined_report_path}")

if __name__ == "__main__":
    main()